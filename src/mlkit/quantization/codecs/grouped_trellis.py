from mlkit.quantization.codecs.registry import codec
from mlkit.quantization.codecs.trellis import decode_trellis


@codec("grouped_trellis", row_parameters=("scales",))
def decode_grouped_trellis(codes, *, scales, group, **parameters):
    weight = decode_trellis(codes, **parameters)
    return (weight.reshape(*scales.shape, group) * scales[..., None]).reshape(weight.shape)
