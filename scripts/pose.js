(function (global) {
    const videoWidth = 640; //screen.width;
    const videoHeight = 480; //screen.height;
    const miniVideoWidth = 160;
    const miniVideoHeight = 120;
    const mini_scale = 0.25;
    var actionCount = 0;
    var lastPair = null;
    var lastActionTime = 0;
    const actionSpanTime = 100;
    const miniDistance = 0;
    const maxDistanceRatio = 0.3;
    const maxDistance = 300; //maxDistanceRatio * videoWidth;
    const leftRightMiniDistance = 10;
    var clearCanvas = true;
    var xOffset = 0;(screen.availWidth - videoWidth)/2;
    var yOffset = 0;(screen.availHeight - videoHeight)/3;

    const guiState = {
        // algorithm: 'multi-pose',
        algorithm: 'single-pose',
        input: {
            mobileNetArchitecture: isMobile() ? '0.50' : '0.75',
            outputStride: 16,
            imageScaleFactor: 0.5,
        },
        singlePoseDetection: {
            minPoseConfidence: 0.5,
            minPartConfidence: 0.1,
        },
        multiPoseDetection: {
            maxPoseDetections: 5,
            minPoseConfidence: 0.15,
            minPartConfidence: 0.1,
            nmsRadius: 30.0,
        },
        output: {
            showVideo: false,
            showSkeleton: false,
            showPoints: true,
            showBoundingBox: false,
        },
        net: null,
    };

    // Pose Dragger
    var PoseDragger = function(){
        this.events = {};

        this.on = function(event, handler){
            this.events[event] = handler;
        };

        this.fire = function(event, data){
            if(!event in this.events){
                throw new Error("");
            }
            this.events[event].apply(this, data);
        };
    };

    var PersonDragger = function(){
        this.left = new PoseDragger();
        this.right = new PoseDragger();
    }; 

    var personDragger1 = new PersonDragger();
    var personDraggers = [personDragger1];

    function showInfo(message) {
        info = document.getElementById("info");
        info.innerHTML = message;
    }

    function isAndroid() {
        return /Android/i.test(navigator.userAgent);
    }

    function isiOS() {
        return /iPhone|iPad|iPod/i.test(navigator.userAgent);
    }

    function isMobile() {
        return isAndroid() || isiOS();
    }

    async function setupCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error(
                'Browser API navigator.mediaDevices.getUserMedia not available');
        }

        const video = document.getElementById('video');
        video.width = videoWidth;
        video.height = videoHeight;

        const mobile = isMobile();
        const stream = await navigator.mediaDevices.getUserMedia({
            'audio': false,
            'video': {
                facingMode: 'user',
                width: mobile ? undefined : videoWidth,
                height: mobile ? undefined : videoHeight,
            },
        });
        video.srcObject = stream;

        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    }

    async function loadVideo() {
        const video = await setupCamera();
        video.play();

        return video;
    }

    function getHand(){
        var radios = document.getElementsByName("hand");
        for(var i = 0; i < radios.length; i++){
            if(radios[i].checked){
                return radios[i].value;
            }
        }
    }

    function detectPoseInRealTime(video, net) {
        const canvas = document.getElementById('output');
        const canvas_mini = document.getElementById('output_mini');
        const ctx = canvas.getContext('2d');
        const ctx_mini = canvas_mini.getContext('2d');
        // since images are being fed from a webcam
        const flipHorizontal = true;

        canvas.width = videoWidth;
        canvas.height = videoHeight;
        canvas_mini.width = miniVideoWidth;
        canvas_mini.height = miniVideoHeight;

        async function poseDetectionFrame() {
            if (guiState.changeToArchitecture) {
                // Important to purge variables and free up GPU memory
                guiState.net.dispose();

                // Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01
                // version
                guiState.net = await posenet.load(+guiState.changeToArchitecture);

                guiState.changeToArchitecture = null;
            }

            // Begin monitoring code for frames per second
            // stats.begin();

            // Scale an image down to a certain factor. Too large of an image will slow
            // down the GPU
            const imageScaleFactor = guiState.input.imageScaleFactor;
            const outputStride = +guiState.input.outputStride;

            let poses = [];
            let minPoseConfidence;
            let minPartConfidence;
            switch (guiState.algorithm) {
                case 'single-pose':
                    const pose = await guiState.net.estimateSinglePose(
                        video, imageScaleFactor, flipHorizontal, outputStride);
                    poses.push(pose);

                    minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
                    minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
                    break;
                case 'multi-pose':
                    poses = await guiState.net.estimateMultiplePoses(
                        video, imageScaleFactor, flipHorizontal, outputStride,
                        guiState.multiPoseDetection.maxPoseDetections,
                        guiState.multiPoseDetection.minPartConfidence,
                        guiState.multiPoseDetection.nmsRadius);

                    minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
                    minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
                    break;
            }

            // ctx.clearRect(0, 0, videoWidth, videoHeight);

            if (guiState.output.showVideo) {
                ctx.save();
                ctx.scale(-1, 1);
                ctx.translate(-videoWidth, 0);
                ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
                ctx.restore();
            }

            // For each pose (i.e. person) detected in an image, loop through the poses
            // and draw the resulting skeleton and keypoints if over certain confidence
            // scores
            poses.forEach(({
                score,
                keypoints
            }) => {
                if (score >= minPoseConfidence) {

                    const now = new Date().getTime();

                    // if(actionCount % 20 == 0){
                    //   ctx.clearRect(0, 0, videoWidth, videoHeight);
                    //   lastKeypoint = null;
                    // }

                    // drawKeypoints(keypoints, minPartConfidence, ctx, scale = 1, radius = 3);

                    // 小窗中绘制识别到的关键点
                    ctx_mini.clearRect(0, 0, canvas_mini.width, canvas_mini.height);
                    drawKeypoints(keypoints, minPartConfidence, ctx_mini, scale = mini_scale, radius = 3);
                    drawSkeleton(keypoints, minPartConfidence, ctx_mini, scale = mini_scale);

                    if (guiState.output.showPoints) {
                        filteredKeypoints = filterKeypoints(["rightWrist", "leftWrist"],
                            keypoints, minPartConfidence);
                        // console.log(JSON.stringify(filteredKeypoints));
                        if (filteredKeypoints.length > 0) {
                            // ctx.clearRect(0, 0, videoWidth, videoHeight);
                            currentPair = splitKeyPoints(filteredKeypoints);
                            if (lastPair == null) {
                                lastPair = currentPair;
                                lastActionTime = now;
                            } else {
                                if (actionCount >= 2) {
                                    ctx.clearRect(0, 0, videoWidth, videoHeight);
                                    actionCount = 0;
                                    // personDragger1.left.fire("startDrag", []);
                                }

                                drawKeypoints(filteredKeypoints, minPartConfidence, ctx, scale = 1,
                                    radius = 10);

                                // left
                                if ("left" == getHand() && lastPair.left != null && currentPair.left != null) {
                                    // let leftDistance = calcDistance(lastPair.left, currentPair.left);
                                    // if (leftDistance > miniDistance && leftDistance < maxDistance) {
                                    if(true){
                                        var dx = currentPair.left.position.x - lastPair.left.position.x,
                                            dy = currentPair.left.position.y - lastPair.left.position.y,
                                            x = currentPair.left.position.x + xOffset, 
                                            y = currentPair.left.position.y + yOffset;

                                        // showInfo("x=" + x + ", y=" + y + ", distance=" + leftDistance);
                                        personDragger1.left.fire("returnValue", [dx, dy, x, y, null, 'left']);
                                        
                                        drawSegment([lastPair.left.position.y,
                                                lastPair.left.position.x
                                            ],
                                            [currentPair.left.position.y,
                                                currentPair.left.position.x
                                            ], "black", 1, ctx);

                                        lastPair.left = currentPair.left;
                                        actionCount++;
                                    }
                                }

                                //right
                                if ("right" == getHand() && lastPair.right != null && currentPair.right != null) {
                                    // let rightDistance = calcDistance(lastPair.right, currentPair.right);
                                    // if (rightDistance > miniDistance && rightDistance < maxDistance) {
                                    if(true){
                                        var dx = lastPair.right.position.x - currentPair.right.position.x,
                                            dy = lastPair.right.position.y - currentPair.right.position.y,
                                            x = lastPair.right.position.x + xOffset, 
                                            y = lastPair.right.position.y + yOffset;

                                        // showInfo("x=" + x + ", y=" + y);
                                        personDragger1.right.fire("returnValue", [dx, dy, x, y, null, 'right']);
                                        drawSegment([lastPair.right.position.y,
                                                lastPair.right.position.x
                                            ],
                                            [currentPair.right.position.y,
                                                currentPair.right.position.x
                                            ], "DeepPink", 1, ctx);

                                        lastPair.right = currentPair.right;
                                        actionCount++;
                                    }
                                }

                                lastActionTime = now;
                            }
                        }
                    }
                    if (guiState.output.showSkeleton) {
                        drawSkeleton(keypoints, minPartConfidence, ctx);
                    }
                    if (guiState.output.showBoundingBox) {
                        drawBoundingBox(keypoints, ctx);
                    }
                }
            });

            // End monitoring code for frames per second
            // stats.end();

            requestAnimationFrame(poseDetectionFrame);
        }

        poseDetectionFrame();
    }

    function calcDistance(point1, point2) {
        if (point1.position.x <= 0 || point1.position.x >= videoWidth) {
            // console.log("x1=" + point1.position.x);
        }
        if (point2.position.x <= 0 || point2.position.x >= videoWidth) {
            // console.log("x2=" + point2.position.x);
        }
        if (point1.position.y <= 0 || point1.position.y >= videoHeight) {
            // console.log("y1=" + point1.position.y);
        }
        if (point2.position.y <= 0 || point2.position.y >= videoHeight) {
            // console.log("y2=" + point2.position.y);
        }
        diff = Math.sqrt(Math.pow(point1.position.x - point2.position.x, 2) +
            Math.pow(point1.position.y - point2.position.y, 2));
        return diff;
    }

    function splitKeyPoints(points) {
        pair = {
            left: null,
            right: null
        };
        for (var i = 0; i < points.length; i++) {
            point = points[i];
            // 左右手是反的
            if (point.part.startsWith("left")) {
                pair.right = point;
            } else if (point.part.startsWith("right")) {
                pair.left = point;
            }
        }

        return pair;
    }

    function filterKeypoints(part_names, keypoints, minPartConfidence, miniDistance=leftRightMiniDistance, maxX = videoWidth, maxY=videoHeight) {
        result = [];
        for (var point_index = 0; point_index < keypoints.length && result.length < part_names.length; point_index++) {
            point = keypoints[point_index]
            for (var name_index = 0; name_index < part_names.length; name_index++) {
                if (part_names[name_index] == point.part){
                    if(point.score > minPartConfidence && 
                        point.position.x <= maxX && point.position.y <= maxY && 
                        point.position.x >= 0 && point.position.y >= 0) {
                        result.push(point);
                    }
                }
            }
        }
        return result;
        
        /* filteredPoints = [];
        if(result.length == 1){
            return result;
        }
        // 过滤距离太近的点
        pcList = permutate_and_combine(result);
        
        for (var i = 0; i < pcList.length; i++) {
            if (pcList[i].length == 2) {
                diff = calcDistance(pcList[i][0], pcList[i][1]);
                if (diff > miniDistance) {
                    filteredPoints.push(pcList[i][0]);
                    filteredPoints.push(pcList[i][1]);
                } else if((pcList[i][0].part == "leftWrist" && pcList[i][1].part == "rightWrist") || 
                    (pcList[i][0].part == "rightWrist" && pcList[i][1].part == "leftWrist")){
                    if(pcList[i][0].part.startsWith(getHand())){
                        filteredPoints.push(pcList[i][0]);
                    }else{
                        filteredPoints.push(pcList[i][1]);
                    }
                } else {
                    if (Math.floor(Math.random() * 10) % 2 == 0) {
                        filteredPoints.push(pcList[i][0]);
                        // console.log("两点距离太近：" + diff + " 丢弃：" + 1);
                    } else {
                        filteredPoints.push(pcList[i][1]);
                        // console.log("两点距离太近：" + diff + " 丢弃：" + 0);
                    }

                }
            }
        }

        return filteredPoints; */
    }

    /**
     * 排列组合
     */
    function permutate_and_combine(arr) {
        result = [];
        if (arr.length > 1) {
            for (i = 0; i < arr.length - 1; i++) {
                for (j = i + 1; j < arr.length; j++) {
                    result.push([arr[i], arr[j]]);
                }
            }
        } else if (arr.length == 1) {
            result.push(arr);
        }

        return result;
    }

    const boundingBoxColor = 'red';
    const lineWidth = 2;

    function toTuple({
        y,
        x
    }) {
        return [y, x];
    }

    function drawPoint(ctx, y, x, r, color) {
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
    }

    /**
     * Draws a line on a canvas, i.e. a joint
     */
    function drawSegment([ay, ax], [by, bx], color, scale, ctx) {
        ctx.beginPath();
        ctx.moveTo(ax * scale, ay * scale);
        ctx.lineTo(bx * scale, by * scale);
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = color;
        ctx.stroke();
    }

    /**
     * Draws a pose skeleton by looking up all adjacent keypoints/joints
     */
    function drawSkeleton(keypoints, minConfidence, ctx, scale = 1, color='cyan') {
        const adjacentKeyPoints = posenet.getAdjacentKeyPoints(keypoints, minConfidence);

        adjacentKeyPoints.forEach(keypoints => {
            drawSegment(toTuple(keypoints[0].position), toTuple(keypoints[1].position), color, scale, ctx);
        });
    }

    /**
     * Draw pose keypoints onto a canvas
     */
    function drawKeypoints(keypoints, minConfidence, ctx, scale = 1, radius = 3) {
        // let ret = {};
        for (let i = 0; i < keypoints.length; i++) {
            const keypoint = keypoints[i];

            if (keypoint.score < minConfidence) {
                continue;
            }

            const {
                y,
                x
            } = keypoint.position;
            let color = 'cyan';
            if (keypoint.part.startsWith("left")) {
                color = "lime";
            } else if (keypoint.part.startsWith("right")) {
                color = "red";
            } else if (keypoint.part == "nose") {
                color = 'yellow';
            }
            let x_ = Math.round(x * scale);
            let y_ = Math.round(y * scale);
            drawPoint(ctx, y_, x_, radius, color);
            // ret[keypoint.part] = {x: x, y:y, x_:x_, y_:y_};
        }
        // console.log(ret);
    }

    /**
     * Draw the bounding box of a pose. For example, for a whole person standing
     * in an image, the bounding box will begin at the nose and extend to one of
     * ankles
     */
    function drawBoundingBox(keypoints, ctx) {
        const boundingBox = posenet.getBoundingBox(keypoints);

        ctx.rect(boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX, boundingBox.maxY - boundingBox.minY);

        ctx.strokeStyle = boundingBoxColor;
        ctx.stroke();
    }

    /**
     * Converts an arary of pixel data into an ImageData object
     */
    async function renderToCanvas(a, ctx) {
        const [height, width] = a.shape;
        const imageData = new ImageData(width, height);

        const data = await a.data();

        for (let i = 0; i < height * width; ++i) {
            const j = i * 4;
            const k = i * 3;

            imageData.data[j + 0] = data[k + 0];
            imageData.data[j + 1] = data[k + 1];
            imageData.data[j + 2] = data[k + 2];
            imageData.data[j + 3] = 255;
        }

        ctx.putImageData(imageData, 0, 0);
    }

    /**
     * Draw an image on a canvas
     */
    function renderImageToCanvas(image, size, canvas) {
        canvas.width = size[0];
        canvas.height = size[1];
        const ctx = canvas.getContext('2d');

        ctx.drawImage(image, 0, 0);
    }

    async function bindPage() {

        // Load the PoseNet model weights with architecture 0.75
        showInfo("正在加载 posenet 模型...");
        const net = await posenet.load(0.75);
        showInfo("posenet 模型加载完毕");
        guiState.net = net;

        // document.getElementById('loading').style.display = 'none';
        document.getElementById('main').style.display = 'block';
        document.getElementById('clear').onclick = function () {
            output = document.getElementById('output');
            ctx = output.getContext('2d');
            ctx.clearRect(0, 0, videoWidth, videoHeight);
        };

        let video;

        try {
            showInfo('正在加载视频设备...');
            video = await loadVideo();
            showInfo('视频设备加载完毕');
        } catch (e) {
            let info = document.getElementById('info');
            info.textContent = '此浏览器不支持视频捕捉或者此设备没有摄像头';
            info.style.display = 'block';
            throw e;
        }

        showInfo("开始检测姿势");
        detectPoseInRealTime(video, net);
    }

    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    // kick off the demo
    bindPage();

    global.personDraggers = personDraggers;
})(this);