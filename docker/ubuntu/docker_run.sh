WS=$1
docker run \
    --rm \
    --name \
    semantic_segmentation \
    -it \
    -v \
    /tmp/.X11-unix:/tmp/.X11-unix \
    -e \
    HOME=$WS \
    --net=host \
    --gpus \
    all \
    -e \
    DISPLAY=$DISPLAY \
    -w \
    $WS \
    -v \
    $HOME:$HOME \
    --device=/dev/dri:/dev/dri \
    -it \
    semantic_segmentation:latest \
    bash