import axios from "axios";

const baseUrl = "this-won't word unless you put your api URL";

function transform(data) {
  return axios.post(baseUrl, data);
}

export { transform };
