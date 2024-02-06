---
name: Bug report
about: Create a report to help us improve
title: ""
labels: "bug"
assignees: ""
---

**Describe the bug**

<!-- A clear and concise description of what the bug is. -->

**To Reproduce**

Script that can be used to send a request that reproduced the behavior (preferably using `grpcurl`):
```
  grpcurl -plaintext -proto proto/generation.proto -d \
    '{
      "model_id": "",
      "requests": [
        {
          "text": "<Text that causes the bad behavior>"
        }
      ],
      "params":{
      }
    }' \
    localhost:8033 fmaas.GenerationService/Generate
```

**Expected behavior**

<!-- A clear and concise description of what you expected to happen. -->

**Screenshots**

<!-- If applicable, add screenshots to help explain your problem. -->

**Environment (please complete the following information):**

- OS/Architecture [e.g. linux amd64]
- Version [e.g. git commit short-hash 5f67482]
- Model [e.g. bigscience/bloom]

**Additional context**

<!-- Add any other context about the problem here. -->
