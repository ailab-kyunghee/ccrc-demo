<template>
  <div>
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
</template>

<script>
export default {
  name: "start-video-practice",
  data() {
    return {
      video: null,
      canvas: null,
      photoSrc: null,
      streaming: false,
      height: 0,
      width: 960
    }
  },
  mounted() {
    this.video = this.$refs.video;
    this.canvas = this.$refs.canvas;
    this.getMediaStream();
    this.clearPhoto();
  },
  methods: {
    getMediaStream() {
      navigator.mediaDevices.getUserMedia(
          {video: true, audio: false}
      ).then((stream) => {
        this.video.srcObject = stream;
        this.video.play();
      }).catch((err) => {
        console.error(`error occurred : ${err}`);
      })
    }, playVideo() {
      if (!this.streaming) {
        this.streaming = true;
        this.height = this.video.videoHeight / this.video.videoWidth * this.width;

        this.video.height = this.height;
        this.video.width = this.width;
        this.canvas.height = this.height;
        this.canvas.width = this.width;
      }
    }, takePhoto() {
      const context = this.canvas.getContext('2d');
      context.translate(this.width, 0);
      context.scale(-1, 1);
      context.drawImage(this.video, 0, 0, this.width, this.height);
      context.setTransform(1, 0, 0, 1, 0, 0); // Reset transformation
      this.photoSrc = this.canvas.toDataURL('image/png');
    }, clearPhoto() {
      const context = this.canvas.getContext('2d');
      context.fillStyle = "#AAA";
      context.fillRect(0, 0, this.width, this.height);
      
      this.photoSrc = null;

    },
    retry() {
      location.reload(true); 
    },
    explain() {
      const imageData = this.canvas.toDataURL('image/png')
      const xhr = new XMLHttpRequest();
      xhr.open('POST', 'http://127.0.0.1:5000/explain', false); // 마지막 인자를 true에서 false로 변경하여 동기적으로 요청합니다.
      xhr.setRequestHeader('Content-Type', 'application/json');
      xhr.send(JSON.stringify({ image_data: imageData }));

      if (xhr.status === 200) {
        console.log(JSON.parse(xhr.responseText.replace(/'/g, '"')));
      } else {
        console.error('Error:', xhr.statusText);
      }
      //   fetch('http://127.0.0.1:5000/explain', {
      //   method: 'POST',
      //   headers: {
      //     'Content-Type': 'application/json',
      //   },
      //   body: JSON.stringify({ image_data: imageData }),
      // })
      // .then(response => console.log(response.text()))
      // .then(data => {
      //   console.log('Success:', data);
      // })
      // .catch(error => {
      //   // 오류 처리
      //   console.error('Error:', error);
      // });

    }
  }
}
</script>

<style scoped>
#video {
  border: 1px solid black;
  box-shadow: 2px 2px 3px black;
  width: 960px;
  height: 720px;
  transform: rotateY(180deg);
  -webkit-transform: rotateY(180deg); /* Safari and Chrome */
  -moz-transform: rotateY(180deg); /* Firefox */
}

#photo {
  border: 1px solid black;
  box-shadow: 2px 2px 3px black;
  width: 960px;
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
</style>