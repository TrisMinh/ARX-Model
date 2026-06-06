# V75 Audit Summary

| Track | Val FIT_sim | Test FIT_sim | Test RMSE | Notes |
| --- | ---: | ---: | ---: | --- |
| Causal history train60 | 72.486 | 71.247 | 0.8375 | Production-safe, selected by validation |
| Causal history refit80 | n/a | 71.308 | 0.8358 | Refit train+validation after selection |
| Diagnostic future actuator | 79.189 | 77.532 | 0.6545 | Not production-safe unless future actuator commands are genuinely known |

Conclusion: the clean causal track improved the previous 70.294% test FIT to about 71.3%. The 75% target is reachable only in the non-causal known-future-actuator diagnostic track.

Top leaderboard rows:

| track | shrink | val_FIT_sim | test_FIT_sim | test_RMSE_sim |
| --- | ---: | ---: | ---: | ---: |
| causal_history_train60 | 1.000 | 72.486 | 71.247 | 0.8375 |
| causal_history_train60 | 1.100 | 72.463 | 71.182 | 0.8394 |
| causal_history_train60 | 0.900 | 72.436 | 71.208 | 0.8387 |
| causal_history_train60 | 0.800 | 72.313 | 71.066 | 0.8428 |
| diagnostic_future_actuator_train60 | 1.100 | 79.189 | 77.532 | 0.6545 |
| diagnostic_future_actuator_train60 | 1.000 | 79.156 | 77.579 | 0.6531 |
| diagnostic_future_actuator_train60 | 0.900 | 78.897 | 77.382 | 0.6588 |
| diagnostic_future_actuator_train60 | 0.800 | 78.419 | 76.938 | 0.6718 |
