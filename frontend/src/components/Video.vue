<template>
  <div v-if="!explained" id="init">
    <div id="contents">
      <div class="camera">
        <video
          v-if="!photoSrc"
          id="video"
          ref="video"
          @canplay="playVideo"
        ></video>
        <img
          v-else
          id="photo"
          :src="photoSrc"
          alt="The screen capture will appear in this box."
        />
      </div>
      <canvas id="canvas" ref="canvas"></canvas>
    </div>
    <div>
      <button v-if="!photoSrc" id="takephoto" @click="takePhoto">
        TAKE PHOTO
      </button>
      <button v-else id="explain" @click="explain">EXPLAIN</button>
      <button v-if="photoSrc" id="retry" @click="retry">RETRY</button>
    </div>
  </div>
  <div v-else id="explained">
    <h1>Explanation</h1>
    <div class="grid-container">
      <div class="grid-item" id="ex1">
        <h3>Neuron No.{{ concepts[1] }}</h3>
        <img :src="imageset[0]" alt="Image" />
        <img :src="imageset[1]" alt="Image" />
        <h3>{{ concepts[0] }}</h3>
      </div>
      <div class="grid-item"></div>
      <div class="grid-item">
        <div id="ex2">
          <h3>Neuron No.{{ concepts[4] }}</h3>
          <img :src="imageset[2]" alt="Image" />
          <img :src="imageset[3]" alt="Image" />
          <h3>{{ concepts[3] }}</h3>
        </div>
      </div>
      <div class="grid-item">
        <div id="ex3">
          <h3>Neuron No.{{ concepts[7] }}</h3>
          <img :src="imageset[4]" alt="Image" />
          <img :src="imageset[5]" alt="Image" />
          <h3>{{ concepts[6] }}</h3>
        </div>
      </div>

      <div class="grid-item">
        <div v-if="explained">
          <img :src="heatmaps[0]" alt="heatmap" />
          <img :src="heatmaps[1]" alt="heatmap" />
          <br />
          <img :src="heatmaps[2]" alt="heatmap" />
          <img :src="heatmaps[3]" alt="heatmap" />
          <h3>{{ concepts[18] }}</h3>
        </div>
      </div>
      <div class="grid-item">
        <div id="ex4">
          <h3>Neuron No.{{ concepts[10] }}</h3>
          <img :src="imageset[6]" alt="Image" />
          <img :src="imageset[7]" alt="Image" />
          <h3>{{ concepts[9] }}</h3>
        </div>
      </div>

      <div class="grid-item">
        <div id="ex5">
          <h3>Neuron No.{{ concepts[13] }}</h3>
          <img :src="imageset[8]" alt="Image" />
          <img :src="imageset[9]" alt="Image" />
          <h3>{{ concepts[12] }}</h3>
        </div>
      </div>

      <div class="grid-item">
        <button v-if="photoSrc" id="retry" @click="retry">RETRY</button>
      </div>

      <div class="grid-item">
        <div id="ex6">
          <h3>Neuron No.{{ concepts[16] }}</h3>
          <img :src="imageset[10]" alt="Image" />
          <img :src="imageset[11]" alt="Image" />
          <h3>{{ concepts[15] }}</h3>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';
export default {
    name: "start-video-practice",
    data() {
        return {
            video: null,
            canvas: null,
            photoSrc: null,
            streaming: false,
            explained: false,
            concepts: null,
            height: 720,
            width: 720,
            imageset: [],
            heatmaps: []
        }
    },
    mounted() {
        this.video = this.$refs.video;
        this.canvas = this.$refs.canvas;
        this.getMediaStream();
        this.clearPhoto();
    },
    methods: {
        getImageUrl(imageFileName) {
            // Flask 서버의 이미지 URL을 반환
            return `http://127.0.0.1:5000/images/${imageFileName}`;
        },
        formatNumber(num) {
            return String(num).padStart(4, '0');

        },
        getMediaStream() {
            navigator.mediaDevices.getUserMedia({
                video: {
                    width: {
                        ideal: 480
                    }, // 원하는 가로 크기 설정
                    height: {
                        ideal: 480
                    } // 원하는 세로 크기 설정
                },
                audio: false
            }).then((stream) => {
                this.video.srcObject = stream;
                this.video.play();
            }).catch((err) => {
                console.error(`error occurred : ${err}`);
            })
        },
        playVideo() {
            if (!this.streaming) {
                this.streaming = true;
                const aspectRatio = this.video.videoWidth / this.video.videoHeight;
                this.height = this.width / aspectRatio;

                this.video.height = this.height;
                this.video.width = this.width;
                this.canvas.height = this.height;
                this.canvas.width = this.width;
            }
        },
        takePhoto() {
            const context = this.canvas.getContext('2d');
            context.translate(this.width, 0);
            context.scale(-1, 1);
            context.drawImage(this.video, 0, 0, this.width, this.height);
            context.setTransform(1, 0, 0, 1, 0, 0); // Reset transformation
            this.photoSrc = this.canvas.toDataURL('image/png');
        },
        clearPhoto() {
            const context = this.canvas.getContext('2d');
            context.fillStyle = "#AAA";
            context.fillRect(0, 0, this.width, this.height);

            this.photoSrc = null;

        },
        retry() {
            location.reload(true);
        },
        sendImageToServer(imageData) {
            // 이미지 데이터를 서버로 전송
            axios.post('http://127.0.0.1:5000/explain', {
                    imageData: imageData
                })
                .then(response => {
                    // 서버로부터 받은 분석 결과를 저장
                    this.analysisResult = response.data;
                    console.log('Analysis result:', this.analysisResult);

                    this.concepts = JSON.parse(response.data.replace(/'/g, '"'));
                    console.log('Concepts:', this.concepts);
                    for (let i = 1; i < this.concepts.length; i += 3) {
                        for (let j = 0; j < this.concepts[i + 1].length; j++) {
                            this.imageset.push(require('../assets/example_val_l4_top2/' + this.formatNumber(this.concepts[i]) + "/" + this.concepts[i + 1][j]))
                        }
                    }
                })
                .catch(error => {
                    console.error('Error while sending image data:', error);
                }).then(() => {
                    const timestamp = new Date().getTime();
                    this.heatmaps.push(`http://127.0.0.1:5000/images/class_att?timestamp=${timestamp}`)
                    this.heatmaps.push(`http://127.0.0.1:5000/images/class_ovr?timestamp=${timestamp}`)
                    this.heatmaps.push(`http://127.0.0.1:5000/images/sample_att?timestamp=${timestamp}`)
                    this.heatmaps.push(`http://127.0.0.1:5000/images/sample_ovr?timestamp=${timestamp}`)

                    this.explained = true;
                });
        },
        explain() {
            const imageData = this.canvas.toDataURL('image/png')
            this.sendImageToServer(imageData)
        }
    }
}
</script>

<style scoped>
#video {
  border: 1px solid black;
  box-shadow: 2px 2px 3px black;
  width: 720px;
  height: 720px;
  transform: rotateY(180deg);
  -webkit-transform: rotateY(180deg);
  /* Safari and Chrome */
  -moz-transform: rotateY(180deg);
  /* Firefox */
}

#photo {
  border: 1px solid black;
  box-shadow: 2px 2px 3px black;
  width: 720px;
  height: 720px;
}

#canvas {
  display: none;
}

.camera {
  width: 1020px;
  display: inline-block;
}

.output {
  width: 1020px;
  display: inline-block;
  vertical-align: top;
}

#takephoto,
#retry,
#explain {
  margin: 30px;
  padding: 20px;
  bottom: 32px;
  background-color: white;
  border: 1px solid gray;
  box-shadow: 0px 0px 1px 2px rgba(0, 0, 0, 0.2);
  font-size: 14px;
  font-family: "Lucida Grande", "Arial", sans-serif;
  color: gray;
}

#takephoto:hover,
#retry:hover,
#explain:hover {
  background-color: #f0f0f0;
  color: black;
}

#takephoto:active,
#retry:active,
#explain:active {
  background-color: #e9e9e9;
  box-shadow: 0px 0px 1px 2px rgba(0, 0, 0, 0.3);
}

#explained img {
  height: 150px;
}

.grid-container {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  align-items: center;
  /* 수직 방향 중앙 정렬 */
  justify-content: center;
  /* 수평 방향 중앙 정렬 */
}

.grid-item {
  padding: 0px;
  text-align: center;
}
</style>
