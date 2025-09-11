# Disable FX graph cache to ensure PT2 compilation happens every time (optional)
export TORCHINDUCTOR_FX_GRAPH_CACHE=0

# Enable debug logging (optional)
export TRITONPARSE_DEBUG=1

# Enable NDJSON output (default)
export TRITONPARSE_NDJSON=1

# Enable gzip compression for trace files (optional)
export TRITON_TRACE_GZIP=1
