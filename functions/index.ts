export default {
  async fetch(request, env) {
    return new Response("Hello from LaunchWing!", {
      headers: { "Content-Type": "text/plain" }
    });
  }
};