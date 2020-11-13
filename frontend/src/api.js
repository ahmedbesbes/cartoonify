import axios from "axios";

const baseUrl =
  "https://tbuxfdm545.execute-api.eu-west-3.amazonaws.com/dev/transform";

function transform(data) {
  return axios.post(baseUrl, data);
}

export { transform };
