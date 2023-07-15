docker image build \
    -t \
    semantic_segmentation:latest \
    --build-arg \
    USER_ID=$(id -u) \
    --build-arg \
    GROUP_ID=$(id -g) \
    .