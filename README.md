# swiftJEPA

Optimized inference for Meta's [V-JEPA 2](https://github.com/facebookresearch/vjepa2)
video encoder.

The goal is a faster inference path for the V-JEPA 2 ViT encoder than the official
PyTorch pipeline delivers.

## Approach

1.  Profile the official pipeline end-to-end and identify the bottlenecks.
2.  Deal with the bottlenecks one at a time against the reference implementation.
3.  Re-measure after change and hold what's reproducible under the benchmarking harness.

## License 

MIT: See [LICENSE](LICENSE) for more details 
Third-party terms, including Meta's MIT license and other licensing agreements 
for the vendored `vjepa2/`, are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
