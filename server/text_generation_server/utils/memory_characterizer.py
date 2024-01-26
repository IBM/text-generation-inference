from text_generation_server.pb import generate_pb2
import numpy as np
import torch
import torch.cuda
import torch.distributed
from scipy.optimize import curve_fit
import gc, os, sys

# Set the memory estimation method to auto, manual or off. If PT2C is used, auto will be forced.
ESTIMATE_MEMORY                   = os.getenv("ESTIMATE_MEMORY", "auto")
assert ESTIMATE_MEMORY in ["auto", "manual", "off"]
# Select the batch size that is used to run the tests. The idea is to make the 
# batch large enough so that the measurement is more accurate, i.e. improve signal to
# noise ratio. If set too large it could prevent the estimator from finding a quadratic
# curve after an initial linear part.
ESTIMATE_MEMORY_BATCH_SIZE        = int(os.getenv("ESTIMATE_MEMORY_BATCH_SIZE", "16"))
# Select the minimum amount of samples that is required to interpolate the quadratic curve
ESTIMATE_MEMORY_MIN_SAMPLES       = int(os.getenv("ESTIMATE_MEMORY_MIN_SAMPLES", "5"))
# Select the shortest sequence length used in the tests. The idea is to make the 
# sequence large enough so that the measurement is more accurate.
ESTIMATE_MEMORY_START_SEQ_LEN     = int(os.getenv("ESTIMATE_MEMORY_START_SEQ_LEN", "100"))
# Select the longest sequence length used in the tests. It is adjusted down if an OOM is detected
ESTIMATE_MEMORY_STOP_SEQ_LEN      = int(os.getenv("ESTIMATE_MEMORY_STOP_SEQ_LEN", "1500"))
# This parameter is used while searching for a linear part. It determines how close the points
# must be to be considered part of a candidate line.
ESTIMATE_MEMORY_FIT_THRESHOLD     = float(os.getenv("ESTIMATE_MEMORY_FIT_THRESHOLD", "0.02"))
# Sets how many tokens should be generated to estimate the memory usage for each new
# token after prefill. Setting this too low will affect precision, setting it too high
# makes the estimation slow.
ESTIMATE_MEMORY_NEW_TOKENS        = int(os.getenv("ESTIMATE_MEMORY_NEW_TOKENS", "50"))
# Sets how many data points we need to estimate the next token memory usage. Since it
# is a line, only 2 should be needed, but if the data is noisy more can be added.
ESTIMATE_MEMORY_NEW_TOKEN_SAMPLES = int(os.getenv("ESTIMATE_MEMORY_NEW_TOKEN_SAMPLES", "2"))


# Considering that each sample is expensive and that absolute precision is not that important,
# we can stop the binary search when the distance between the last two samples becomes smaller
# than this constant.
STOPPING_DISTANCE = 10

class MemoryScalingModel:
    def __init__(self, free_memory, safety_margin, linear_fit_params, quadratic_fit_params, next_token_params):
        self.free_memory = free_memory
        self.safety_margin = safety_margin
        self.linear_fit_params = linear_fit_params
        self.quadratic_fit_params = quadratic_fit_params
        self.next_token_params = next_token_params
        self.weight_limit = np.floor((1.0 - self.safety_margin / 100.0) * self.free_memory).astype(np.uint64)

    def as_pb(self):
        return generate_pb2.MemoryScalingModel(
            weight_limit=self.weight_limit,
            prefill_linear_coef0=self.linear_fit_params[0],
            prefill_quadratic_coef0=self.quadratic_fit_params[0],
            prefill_quadratic_coef1=self.quadratic_fit_params[1],
            nexttoken_linear_coef0=self.next_token_params[0],
            nexttoken_linear_coef1=self.next_token_params[1],
        )

    def __prefill_memory_usage(self, batch_size, input_len):
        out1 = batch_size * self.linear_fit_params[0] * input_len
        bs_seq = input_len * batch_size
        out2 = self.quadratic_fit_params[0] * bs_seq + input_len * self.quadratic_fit_params[1] * bs_seq
        return np.maximum(out1, out2)

    def __nt_memory_usage(self, batch_size, input_len, output_len):
        return batch_size * self.next_token_params[0] * input_len + batch_size * self.next_token_params[1] * output_len

    def max_input_len_for_prefill(self, batch_size, max_input_len):
        x = np.arange(1, 1+max_input_len)
        mem_usage = self.__prefill_memory_usage(batch_size, x)
        ind = np.argwhere(mem_usage < self.weight_limit)[-1][0]
        return x[ind]

    def max_input_len_for_nt(self, batch_size, output_len, max_input_len):
        x = np.arange(1, 1+max_input_len)
        mem_usage = self.__nt_memory_usage(batch_size, x, output_len)
        ind = np.argwhere(mem_usage < self.weight_limit)[-1][0]
        return np.minimum(x[ind], self.max_input_len_for_prefill(batch_size, max_input_len))

    def max_output_len_for_nt(self, batch_size, input_len, max_output_len):
        x = np.arange(1, 1+max_output_len)
        mem_usage = self.__nt_memory_usage(batch_size, input_len, x)
        ind = np.argwhere(mem_usage < self.weight_limit)[-1][0]
        return x[ind]
    
    @classmethod
    def disabled(cls):
        return cls(sys.maxsize, 0, [0], [0, 0], [0, 0])

    @classmethod
    def manual_quadratic(cls, safety_margin: int, max_seq_len: int, max_batch_size: int):
        # In this mode the size limit for batches is not given by an measurement
        # of the hardware capacity and the scaling of memory usage w.r.t.
        # sequence length and batch size. The total token capacity here is given by
        # a percentage of MAX_SEQUENCE_LENGTH^2 * BATCH_SIZE in prefill and
        # MAX_SEQUENCE_LENGTH * BATCH_SIZE in the next token phase.
        # The result of evaluating the estimation function is a percentage of how much
        # of the capacity is used. Therefore the "free_memory" in this mode is 100%.
        # The percentages here are numbers from 0 to 100 rather than 0.0 to 1.0 because
        # the variables are integers.

        TOTAL_MEMORY = 100 # 100%
        assert 0 <= safety_margin <= 100
        P = (100.0 - safety_margin) / 100.0
        linear_param = 100.0 / (P * max_seq_len*max_batch_size)
        quadratic_param = 100.0 / (P * max_seq_len*max_seq_len*max_batch_size)
        return cls(TOTAL_MEMORY, safety_margin, [0], [0, quadratic_param], [linear_param, linear_param])

    @classmethod
    def manual_linear(cls, safety_margin: int, max_seq_len: int, max_batch_size: int):
        # In this mode the size limit for batches is not given by an measurement
        # of the hardware capacity and the scaling of memory usage w.r.t. sequence
        # length and batch size. The total token capacity here is given by a percentage
        # of MAX_SEQUENCE_LENGTH * BATCH_SIZE in prefill and in the next token phase.
        # The result of evaluating the estimation function is a percentage of how much
        # of the capacity is used. Therefore the "free_memory" in this mode is 100%.
        # The percentages here are numbers from 0 to 100 rather than 0.0 to 1.0 because
        # the variables are integers.

        TOTAL_MEMORY = 100 # 100%
        assert 0 <= safety_margin <= 100
        P = (100.0 - safety_margin) / 100.0
        linear_param = 100.0 / (P * max_seq_len*max_batch_size)
        return cls(TOTAL_MEMORY, safety_margin, [linear_param], [0, 0], [linear_param, linear_param])




class Estimator:
    def __init__(
        self,
        model: 'Model',
        batch_size: int,
        min_samples: int,
        start_seq_len: int,
        stop_seq_len: int,
        fit_threshold: float,
        new_tokens: int,
        new_token_samples: int,
        safety_margin: int,
    ) -> None:
        assert stop_seq_len > start_seq_len > 0
        assert min_samples > 1
        self.model = model
        self.batch_size = batch_size
        self.min_samples = min_samples
        self.start_seq_len = start_seq_len
        self.stop_seq_len = stop_seq_len
        self.fit_threshold = fit_threshold
        self.X = np.array([[batch_size], [0]], dtype=np.int64)
        self.Y = np.zeros(1, dtype=np.int64)
        self.baseline = 0
        self.free_memory = 0
        self.linear_fit_params = [0.0]
        self.quadratic_fit_params = [0.0, 0.0]
        self.inflection_point = 0
        self.nt_X = np.array([[batch_size], [0], [0]], dtype=np.int64)
        self.nt_Y = np.zeros(1, dtype=np.int64)
        self.next_token_params = [0.0, 0.0]
        self.max_new_tokens = new_tokens
        self.remaining_new_token_samples = new_token_samples
        self.enable_nt_sampling = False
        self.nt_upper_input_limit = int(0.75 * self.stop_seq_len)  # testing above this range is expensive
        self.safety_margin = safety_margin

    @classmethod
    def build_from_env(cls, model: 'Model', safety_margin: int):
        return cls(
                model,
                ESTIMATE_MEMORY_BATCH_SIZE,
                ESTIMATE_MEMORY_MIN_SAMPLES,
                ESTIMATE_MEMORY_START_SEQ_LEN,
                ESTIMATE_MEMORY_STOP_SEQ_LEN,
                ESTIMATE_MEMORY_FIT_THRESHOLD,
                ESTIMATE_MEMORY_NEW_TOKENS,
                ESTIMATE_MEMORY_NEW_TOKEN_SAMPLES,
                safety_margin,
            )

    @staticmethod
    def __quadratic_f(x, a, b):
        batch, seq = x
        return a*batch*seq + b*batch*seq**2

    @staticmethod
    def __linear_f(x, a):
        batch, seq = x
        return a*batch*seq
    
    @staticmethod
    def __next_token_model(X, a, b):
        batch, in_seq, out_seq = X
        return a*batch*in_seq + b*batch*out_seq
    
    @staticmethod
    def __generate_batch(model, batch_size: int, in_text: str, in_tokens: int, num_new_tokens: int):
        request = generate_pb2.PrefillRequest(
            batch=generate_pb2.Batch(
                id=0,
                requests=[
                    generate_pb2.Request(
                        id=i, inputs=in_text, input_length=in_tokens, truncate=True, max_output_length=num_new_tokens
                    ) for i in range(batch_size)
                ]
            )
        )

        batch, _ = model.batch_type.from_pb(
            request.batch,
            tokenizer=model.tokenizer,
            device=model.device,
            dtype=model.dtype,
            embeddings_lookup=model.word_embeddings,
            prefix_cache=model.prefix_cache,
            use_position_ids=model.use_position_ids,
        )
        return batch

    def _sort_samples(self):
        sorted_ind = self.X[1].argsort()
        self.X = self.X[:, sorted_ind]
        self.Y = self.Y[sorted_ind]

    def needs_nt(self):
        return self.remaining_new_token_samples > 0 and self.enable_nt_sampling

    def _run_prefill_test(self, seq_length, min_max_tokens):
        try:
            gc.collect()
            torch.cuda.reset_peak_memory_stats(self.model.device)
            input_text = "test " * 10_000
            batch = Estimator.__generate_batch(
                self.model,
                batch_size=self.batch_size,
                in_text=input_text,
                in_tokens=seq_length,
                num_new_tokens=min_max_tokens,
            )
            self.model.generate_token(batch, first=True, for_concat=False)
            batch = self.model.batch_type.concatenate([batch])
            mem_used = torch.cuda.max_memory_allocated(self.model.device)
            return False, mem_used, batch
        except torch.cuda.OutOfMemoryError: # type: ignore
            return True, None, None

    def _run_next_token_test(self, batch, out_seq):
        ret = []
        try:
            for _ in range(1,out_seq):
                gc.collect()
                torch.cuda.reset_peak_memory_stats(self.model.device)
                self.model.generate_token(batch)
                batch = self.model.batch_type.concatenate([batch])
                mem_used = torch.cuda.max_memory_allocated(self.model.device)
                ret.append(mem_used)
        except torch.cuda.OutOfMemoryError: # type: ignore
            pass

        return ret

    def sample_next_token(self, batch, input_seq_len):
        if input_seq_len >= self.nt_upper_input_limit:
            return

        results = self._run_next_token_test(batch, self.max_new_tokens)

        if len(results) < (self.max_new_tokens-1):
            print(f"got less results than expected {len(results)=}, {self.max_new_tokens=}")
            self.nt_upper_input_limit = input_seq_len
            return
        
        for request, result in enumerate(results):
            out_seq = request + 1
                                
            y = result - self.baseline

            self.nt_X = np.append(self.nt_X, [[self.batch_size], [input_seq_len], [out_seq]], axis=1)
            self.nt_Y = np.append(self.nt_Y, y)
        self.remaining_new_token_samples -= 1

    def take_n_samples(self, n, start, stop):
        x_data = np.array([[], []], dtype=np.int64)
        y_data = np.array([], dtype=np.int64)
        stride = int((stop-start)/n)
        for s in range(start, stop, stride):

            tokens = self.max_new_tokens if self.needs_nt() else 1

            _, mem_allocated, batch = self._run_prefill_test(s, tokens)
            x_data = np.append(x_data, [[self.batch_size], [s]], axis=1)
            y_data = np.append(y_data, mem_allocated - self.baseline)
            if self.needs_nt():
                self.sample_next_token(batch, s)
            del batch
        return x_data, y_data

    def __reduce_error(self, error):
        if self.model.engine.world_size == 1:
            return error

        error_ = torch.tensor([error], device=self.model.device)
        torch.distributed.all_reduce(
            error_,
            op=torch.distributed.ReduceOp.MAX,
            group=self.model.engine.process_group,
        )
        return error_[0]

    def find_baseline(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.model.device)
        self.baseline = torch.cuda.max_memory_allocated(self.model.device)
        self.free_memory, _ = torch.cuda.mem_get_info(self.model.device)
        print("Baseline: %d, Free memory: %d" % (self.baseline, self.free_memory))

    def find_upper_bound(self):
        print("Validating the upper bound")
        # find upper bound for the sequence length that doesn't cause OOM
        lower = self.start_seq_len
        upper = s = self.stop_seq_len
        while (upper - lower) > STOPPING_DISTANCE:
           
            is_oom, mem_allocated, _ = self._run_prefill_test(s, 1)

            if is_oom:
                upper = s
                s = int((s + lower) / 2)
            else:
                self.X = np.append(self.X, [[self.batch_size], [s]], axis=1)
                self.Y = np.append(self.Y, mem_allocated - self.baseline)
                lower = s
                s = int((s + upper) / 2)

        if self.Y.size > 1:
            self.stop_seq_len = lower
            return True
        return False
    
    def find_linear_part(self):
        print("Looking for the linear part")

        # The first sample is the lowest one that didn't cause an OOM error
        # Testing with shorter sequence lengths is less expensive
        upper_y = self.Y[1]
        upper_x = self.X[1, 1]
        lower_x = min(self.start_seq_len, upper_x)

        longest_linear = 0

        while lower_x <= (upper_x-1):

            gradient = upper_y / upper_x
            target = int((lower_x + upper_x)/2)

            x_samples, y_samples = self.take_n_samples(1, target, target+1)
            
            self.X = np.append(self.X, x_samples, axis=1)
            self.Y = np.append(self.Y, y_samples)

            measured  = y_samples[0]
            projected = gradient * target

            error = self.__reduce_error(
                np.absolute(measured-projected) / measured
            )

            if error <= self.fit_threshold:
                longest_linear = upper_x
                break
            else:
                upper_x = target
                upper_y = measured

        self._sort_samples()

        if longest_linear == 0:
            print("Warning: couldn't find a linear part")
            return longest_linear

        last_index = np.where(self.X[1] == longest_linear)[0][0]

        # Check whether we have existing samples that are linear
        higher_indices = np.where(self.X[1] > longest_linear)[0]
        if higher_indices.size > 0:
            for i in range(higher_indices[0], self.X.shape[1]):
                new_gradient = self.Y[i]/self.X[1, i]
                error = self.__reduce_error(
                    np.absolute(new_gradient-gradient)/gradient
                )
                if error <= self.fit_threshold:
                    longest_linear = self.X[1, i]
                    last_index = i
                else:
                    break

        self.linear_fit_params, _ = curve_fit(
            Estimator.__linear_f, self.X[:, :last_index], self.Y[:last_index], bounds=([0], [np.inf])
        )
        return longest_linear

    def find_inflection_point(self, longest_linear):
        print("Looking for the inflection point")
        lower = longest_linear
        upper = self.stop_seq_len

        s = int((lower + upper)/2)

        while (upper - lower) > STOPPING_DISTANCE:
            x_samples, y_samples = self.take_n_samples(1, s, s+1)
            self.X = np.append(self.X, x_samples, axis=1)
            self.Y = np.append(self.Y, y_samples)

            error = self.__reduce_error(
                np.absolute((y_samples-Estimator.__linear_f(x_samples, *self.linear_fit_params)) / y_samples)[0]
            )

            if error <= self.fit_threshold:
                lower = s
            else:
                upper = s
            s = int((lower + upper)/2)
        self._sort_samples()

        return self.stop_seq_len if (self.stop_seq_len - s) < STOPPING_DISTANCE else s

    def estimate_quadratic(self, start):
        print(f"Estimating quadratic part, starting at {start}")
        X = self.X[:, self.X[1, :] > start]

        existing_samples  = X.shape[1]
        remaining_samples = self.min_samples - existing_samples

        Y = self.Y[-existing_samples:]

        if remaining_samples > 0:
            x_samples, y_samples = self.take_n_samples(remaining_samples, start + 1, self.stop_seq_len)
            X = np.append(X, x_samples, axis=1)
            Y = np.append(Y, y_samples)

        self.quadratic_fit_params, _ = curve_fit(Estimator.__quadratic_f, X, Y, bounds=([0, 0], [np.inf, np.inf]))

    def init_nt_sampling(self):
        self.enable_nt_sampling = True
        self.nt_upper_input_limit = int(0.75 * self.stop_seq_len)

    def estimate_nt(self):
        if self.needs_nt():
            if self.nt_upper_input_limit - self.start_seq_len < self.remaining_new_token_samples:
                raise RuntimeError(
                    f"Unable to fit next token memory model; try using smaller value"
                    f" for ESTIMATE_MEMORY_BATCH_SIZE (currently set to {self.batch_size})"
                )
            self.take_n_samples(self.remaining_new_token_samples, self.start_seq_len, self.nt_upper_input_limit)

        self.next_token_params, _ = curve_fit(
            Estimator.__next_token_model, self.nt_X, self.nt_Y, bounds=([0, 0], [np.inf, np.inf])
        )

    def run(self):

        if not torch.cuda.is_available():
            print("Disabling memory usage estimation for CPU-only execution")
            return MemoryScalingModel.disabled()

        self.find_baseline()

        found = self.find_upper_bound()
        self.init_nt_sampling()
        
        if not found:
            raise Exception(
                f"Couldn't find prefill sequence in range {self.start_seq_len}-{self.stop_seq_len}"
                f" that doesn't run OOM"
            )
        
        longest_linear = self.find_linear_part()
        assert 0 <= longest_linear <= self.stop_seq_len

        self.inflection_point = longest_linear

        # If the curve is neither all linear nor all quadratic, find the inflection point
        if self.start_seq_len < longest_linear < self.stop_seq_len:
            self.inflection_point = self.find_inflection_point(longest_linear)
        
        if self.inflection_point != self.stop_seq_len:
            self.estimate_quadratic(self.inflection_point)

        self.estimate_nt()

        print(">> fitted model:")
        print(">> free_memory:          ", self.free_memory)
        print(">> linear_fit_params:    ", self.linear_fit_params)
        print(">> quadratic_fit_params: ", self.quadratic_fit_params)
        print(">> next_token_param:     ", self.next_token_params)

        return MemoryScalingModel(
            self.free_memory,
            self.safety_margin,
            self.linear_fit_params,
            self.quadratic_fit_params,
            self.next_token_params,
        )
