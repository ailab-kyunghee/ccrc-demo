<template>
  <Video msg="제목" />
</template>

<script>
import Video from './components/Video.vue'

export default {
  name: 'App',
  components: {
    Video
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
