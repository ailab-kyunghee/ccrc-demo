<template>
  <div class="page-container">
    <!-- Loading overlay (spinner) -->
    <div v-if="loading" class="loading-overlay">
      <div class="spinner"></div>
    </div>

    <h1>Neural Reasoning Demo</h1>

    <div class="container-row">

      <!-- Input Section -->
      <div class="input-section column">
        <h2>Input</h2>
        <p>Please select an image for explanation.</p>
        <div class="image-upload-box image-container">
          <img :src="photoSrc || sampleImage" alt="Selected Image" />
        </div>
        <div>
          <input type="file" accept="image/*" @change="handleFileUpload" class="file-upload" />
          <div class="button_div_style">
            <button id="explain" @click="explain">Explain</button>
            <button>Retry</button>
          </div>
        </div>
      </div>

      <!-- Prediction Section -->
      <div class="prediction-section column">
        <h2>Prediction:</h2>
        <h2>{{ prediction1.headding || "Awaiting Prediction..."}}</h2>
        <div class="image-container">
          <img :src="prediction1.image || sampleImage" alt="Prediction 1 Image" />
        </div>
      </div>

      <!-- Explanation Section -->
      <div class="explanation-section column">
        <h2>Explanation</h2>
        <h2>Patient Case</h2>
        <div class="image-container">
          <img :src="explanation1.patientCase || sampleImage" alt="Patient Case 1" />
        </div>
        <h3>Report</h3>
        <p class="report-text">{{ prediction1.report }}</p>
        <h3>Concepts</h3>
        <p class="report-text">{{ concept1 }}</p>
        <p class="report-text">{{ concept2 }}</p>
      </div>

    </div>
  </div>
</template>

<script>
import axios from 'axios';
export default {
  data() {
    return {
      photoSrc: null,
      loading: false,
      sampleImage: 'http://127.0.0.1:5000/images/default-image.jpg',
      prediction1: {
        image: null,
        report: null
      },
      explanation1: {
        generalCase: null,
        patientCase: null
      }
    };
  },
  methods: {
    handleFileUpload(event) {
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          this.photoSrc = e.target.result; // Display the selected image
        };
        reader.readAsDataURL(file);
        this.selectedFile = file;
      }
    },
    async explain() {
      if (!this.selectedFile) {
        alert('Please select an image file first.');
        return;
      }

      const formData = new FormData();
      formData.append('image', this.selectedFile);

      // Start loading before the API call
      this.loading = true;

      try {
        const response = await axios.post('http://127.0.0.1:5000/explain', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });

        this.loadPredictionAndExplanation(response.data);
      } catch (error) {
        console.error('Error during the request:', error);
        alert('An error occurred while processing the image.');
      }
      finally {
        // Stop loading after the API call
        this.loading = false;
      }
    },
    loadPredictionAndExplanation(data) {
      this.prediction1.image = data["input_image"];
      this.prediction1.report = data.report;
      this.prediction1.headding = data.pred;
      this.explanation1.patientCase = data["exp-pc-1"];
      this.concept1 = data["concept1"];
      this.concept2 = data["concept2"];
    }
  }
};
</script>

<style scoped>
body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 0;
}

.page-container {
  text-align: center;
  margin: 20px auto;
  max-width: 1200px;
}

h1 {
  text-align: center;
  margin-bottom: 40px;
}

.container-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin: 0 auto;
  width: 90%;
}

.column {
  width: 30%;
  text-align: center;
}

.image-container {
  width: 250px;
  height: 250px;
  background-color: #f0f0f0;
  border: 1px solid #ccc;
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 20px auto;
}

.image-container img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.file-upload {
  margin-top: 10px;
}

button {
  padding: 10px 20px;
  margin: 10px 5px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

button:hover {
  background-color: #0056b3;
}

button:active {
  background-color: #00408a;
}

button+button {
  margin-left: 10px;
}

.button_div_style {
  display: flex;
  justify-content: center;
  margin-top: 20px;
}

.report-text {
  margin-top: 15px;
  text-align: center;
}

h2, h3 {
  text-align: center;
  margin-bottom: 20px;
}

/* Styling for the loading overlay */
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.spinner {
  width: 50px;
  height: 50px;
  border: 5px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: spin 1s ease infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

</style>
