NCLASSES = 4

# Maps image sizes to pre-trained models to use on those images.
pretrained_map = {
    256: {
        "BiT": "https://tfhub.dev/google/bit/m-r101x1/1",
        "ResNetV2-101": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4",
        "InceptionV3": "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
        "EfficientNetB3": "https://tfhub.dev/google/efficientnet/b3/feature-vector/1",
    },
    512: {
        "BiT": "https://tfhub.dev/google/bit/m-r101x1/1",
        "EfficientNetB6": "https://tfhub.dev/google/efficientnet/b6/feature-vector/1",
    },
}
