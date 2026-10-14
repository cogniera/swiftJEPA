#Author : Paarth Sharma
#File Name : README.md
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : Preserve an official pretrained fp16 quality failure before latency timing.

The official eager ViT-L encoder at `ca6d01a` was compared under fp16 autocast
against its untouched fp32 weights on seven raw clips, each repeated. The
minimum cosine is 0.9977144501 on moving color bars. Mid-gray also fails 0.999.
The source config and complete environment are recorded in `eager_1/run.json`.

This attempt exited before warmup and timing. No latency samples exist and no
performance result can be inferred. The required official comparator will be
measured with this quality finding disclosed. The engine threshold stays 0.999.
