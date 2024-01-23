"""

This script can be used to help verify correctness of models running in TGIS
with dynamic batching.

False negatives are still possible since there is some amount of inconsistency expected due to
the fixed precision floating point operations, for example in float16 and especially bfloat16.

"""
import concurrent.futures
import time

import grpc
import generation_pb2
import generation_pb2_grpc


class col:
    RED = '\033[31m'
    ENDC = '\033[m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'


if __name__ == "__main__":
    channel = grpc.insecure_channel("localhost:8033")
    stub = generation_pb2_grpc.GenerationServiceStub(channel)

    # 1) Common input + output length batch
    #   -- Based on this determine unique stop sequence
    # 2) Variable input length, common output length batch
    # 3) Common input length, variable output length batch (using stop sequence from 1)
    # 4) Concatenation test, long, with short concurrently after tiny delay

    model_id = "unused"

    out_length = 80  # should be at least 50
    long_text = (
        "The core components include: a studio for new foundation models, generative AI and machine learning; "
        "a fit-for-purpose data store built on an open data lakehouse architecture; and a  toolkit, to accelerate "
        "AI workflows that are built with responsibility, transparency and explainability."
    )
    short_text = "watsonx is a generative AI and data platform with a set of AI assistants."

    # Get token counts for inputs
    tresp = stub.Tokenize(generation_pb2.BatchedTokenizeRequest(
        model_id=model_id,
        requests=[generation_pb2.TokenizeRequest(text=text) for text in (long_text, short_text)],
        return_tokens=False,
    ))

    long_tokens, short_tokens = (tr.token_count for tr in tresp.responses)

    print(f"token counts: short={short_tokens}, long={long_tokens}")

    assert 15 < short_tokens < long_tokens

    truncate_to = short_tokens - 10

    def send_requests(requests, truncate_to=0, stop_seqs=None, out_length=out_length, min_out_length=None):
        if min_out_length is None:
            min_out_length = out_length

        return stub.Generate(generation_pb2.BatchedGenerationRequest(
            model_id=model_id,
            requests=requests,
            params=generation_pb2.Parameters(
                truncate_input_tokens=truncate_to,
                stopping=generation_pb2.StoppingCriteria(
                    min_new_tokens=min_out_length,
                    max_new_tokens=out_length,
                    stop_sequences=stop_seqs,
                ),
            ),
        ))

    def log_result(batched_results, individual_results):
        matches = [x == y for (x, y) in zip(batched_results, individual_results)]
        if all(matches):
            print(col.GREEN + "PASS" + col.ENDC)
        else:
            print(f"{col.RED}FAIL{col.ENDC}: {matches.count(False)}/{len(matches)} mismatches: {matches}")
            #TODO improve how these diffs are printed
            print(f"BATCHED: {batched_results}")
            print(f"SINGLE : {individual_results}")


    #TODO first do single request consistency

    #TODO can add a test here to check invariance to front-padding


    greqs = [generation_pb2.GenerationRequest(text=text) for text in (long_text, short_text)]


    ### Test 1 #################################################################################
    ############################################################################################
    test_name = "Common input and output sizes (tests basic batching)"
    print(f"\n{test_name}")
    batched_1 = [gr.text for gr in send_requests(greqs, truncate_to).responses]
    individual_1 = [send_requests([gr], truncate_to).responses[0].text for gr in greqs]
    log_result(batched_1, individual_1)


    ### Test 2 #################################################################################
    ############################################################################################
    test_name = "Variable input lengths, common output length (tests padded batch)"
    print(f"\n{test_name}")
    batched_2 = [gr.text for gr in send_requests(greqs).responses]
    individual_2 = [send_requests([gr]).responses[0].text for gr in greqs]
    log_result(batched_2, individual_2)


    ### Test 3 #################################################################################
    ############################################################################################
    # Find stop seq - a string in one of the output sequences that isn't in the other
    ss = None
    for i in range(20, out_length - 5):
        ss = batched_1[0][i:i+5]
        if ss not in batched_1[1]:
            break

    if ss is None:
        print("\ncouldn't find stop seq to use for variable output length test")
    else:
        test_name = "Common input length, variable output lengths (tests batch pruning)"
        print(f"\n{test_name}")
        print(f"Using stop-sequence '{ss}'")
        batched_3 = [gr.text for gr in send_requests(greqs, truncate_to, [ss], min_out_length=1).responses]
        individual_3 = [
            send_requests([gr], truncate_to, [ss], min_out_length=1).responses[0].text for gr in greqs
        ]
        log_result(batched_3, individual_3)


    ### Test 4 #################################################################################
    ############################################################################################
    test_name = "Short output interrupting long output (tests batch concatenation)"
    print(f"\n{test_name}")
    long_input_req, short_input_req = greqs
    individual_4 = [
        send_requests([short_input_req], out_length=100).responses[0].text,
        send_requests([long_input_req], out_length=10).responses[0].text,
    ]

    with concurrent.futures.ThreadPoolExecutor(1) as executor:
        future = executor.submit(send_requests, [short_input_req], out_length=100)
        time.sleep(0.25)
        short_resp = send_requests([long_input_req], out_length=10).responses[0]
        long_resp = future.result().responses[0]
        batched_4 = [long_resp.text, short_resp.text]
    log_result(batched_4, individual_4)
