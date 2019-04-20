const faceREPL = (function(){

    const BBOX_HAIR_RATIO = 0.5;

    async function getFaceDetector(){
        // thank you, https://github.com/justadudewhohacks/face-api.js
        const modelURL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js-models@master/tiny_face_detector/tiny_face_detector_model-weights_manifest.json';
        await faceapi.loadTinyFaceDetectorModel(modelURL);
        const net = (image, inputSize=320) => faceapi.tinyFaceDetector(
            image, faceapi.TinyFaceDetectorOptions({inputSize})
        );
        return net;
    }

    async function getVideoStream(vidSize){
        if(navigator.mediaDevices.getUserMedia){
            return await navigator.mediaDevices.getUserMedia({audio: false,video:
                {width: vidSize[0], height: vidSize[1], facingMode: 'user'}})
                .then(stream => {
                    let video = document.createElement('video');
                    video.setAttribute('autoplay', true);
                    video.srcObject = stream;
                    video.getFrame = () => faceapi.createCanvasFromMedia(video);
                    return video;
                })
                .catch(e => { console.log("Please accept video"); });
        }
    }

    const cropPatches = (sourceCanvas, netOutput) => netOutput.map(({_box}) => {
        const {_x, _y, _width, _height} = _box;
        const subCanvas = document.createElement('canvas');
        const subContext = subCanvas.getContext('2d');
        subCanvas.width = _width;
        subCanvas.height = _height;

        subContext.drawImage(sourceCanvas, _x, _y, _width, _height, 0, 0, _width, _height);
        return subCanvas;
    })

    function patchesOntoFrame(frame, patches, bboxes){
        const frameContext = frame.getContext('2d');
        patches.forEach((patch, i) => {
            const {_x, _y, _width, _height} = bboxes[i]._box;
            frameContext.drawImage(patch, 0,0, _width, _height, _x, _y, _width, _height)
        });
        return frame;
    }

    const tfImageProcesser = tfFunc => async (img) => {
        const pixels = tf.browser.fromPixels(img);
        const newTensor = tfFunc(tf.div(pixels, 255));
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        return await tf.browser.toPixels(newTensor, canvas).then(() => canvas)
    }

    async function iterate(net, video, imgPatchProcess, frameCallback){
        const frame = video.getFrame();
        net(frame).then(boxes => {
            boxes.forEach(({_box}, i) => {
                const orig_y = _box._y;
                boxes[i]._box._y = Math.max(0, _box._y - (_box._height * BBOX_HAIR_RATIO));
                boxes[i]._box._height = _box.height + orig_y - boxes[i]._box._y;
            })
            let patches = cropPatches(frame, boxes).map(imgPatchProcess);
            Promise.all(patches).then(patches => {
                frameCallback(patchesOntoFrame(frame, patches, boxes));
            });
        })
    }

    /**
     * Runs a webcam REPL
     * 
     * @param patchProcess {function} processes a patch of an image (as a tensor)
     * @param frameCallback {function} is given the processed frame
     * @param fps {number} frames per second
     * @param vidSize {number[]} size of the video
     * @returns interval
     */
    async function faceREPL(patchProcess, frameCallback, fps=20, vidSize=[300, 300]){
        return await getFaceDetector().then(net => getVideoStream(vidSize).then(video =>
                setInterval(() => iterate(net, video, tfImageProcesser(patchProcess), frameCallback), 1000 / fps)
        ))
    }

    return faceREPL;
})();