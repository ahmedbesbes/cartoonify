import React, { useState, useEffect } from "react";
import { makeStyles } from "@material-ui/core/styles";

import Box from "@material-ui/core/Box";
import Chip from "@material-ui/core/Chip";
import StyleSelector from "../components/StyleSelector.js";
import Backdrop from "@material-ui/core/Backdrop";
import DownloadIcon from "@material-ui/icons/GetApp";

import ImageUploader from "react-images-upload";
import ReactCompareImage from "react-compare-image";
import * as loadImage from "blueimp-load-image";

import GridLoader from "react-spinners/GridLoader";
import beforePlaceholder from "../images/before.jpg";
import afterPlaceholder from "../images/after.jpg";
import logo from "../images/logo.svg";

import { triggerBase64Download } from "react-base64-downloader";
import { transform } from "../api.js";
import { toDataUrl } from "../utils.js";

const LOAD_SIZE = 450;
const WIDTH = 400;

const useStyles = makeStyles((theme) => ({
  formControl: {
    width: "70%",
    margin: theme.spacing(1),
  },
  holder: {
    width: WIDTH,
    maxWidth: 350,
    margin: 10,
  },
  chip: {
    width: 200,
    backgroundColor: "#e63946",
    margin: 10,
    fontWeight: "bold",
  },
  logo: {
    width: WIDTH * 0.8,
    marginTop: 10,
    marginLeft: 20,
  },
}));

export default function Home() {
  const [before, setBefore] = useState("");
  const [after, setAfter] = useState("");
  const [percentage, setPercentage] = useState(0.5);
  const [modelID, setModelID] = useState(0);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    toDataUrl(beforePlaceholder, (base64) => {
      setBefore(base64);
    });
  }, []);

  useEffect(() => {
    toDataUrl(afterPlaceholder, (base64) => {
      setAfter(base64);
    });
  }, []);

  const classes = useStyles();
  return (
    <Box align="center">
      <div style={{ textAlign: "center", width: "100%" }}>
        <img src={logo} className={classes.logo} />
      </div>
      <div className={classes.holder}>
        <StyleSelector
          modelID={modelID}
          setModelID={setModelID}
          setPercentage={setPercentage}
          setOpen={setOpen}
          before={before}
          LOAD_SIZE={LOAD_SIZE}
          setAfter={setAfter}
        />

        <ImageUploader
          singleImage
          buttonText="Choose images"
          onChange={(pictureFiles, pictureDataURL) => {
            setOpen(true);
            setPercentage(1);

            loadImage(
              pictureDataURL[0],
              (cnv) => {
                setBefore(cnv.toDataURL());
                setAfter(cnv.toDataURL());
                const data = {
                  image: cnv.toDataURL(),
                  model_id: modelID,
                  load_size: LOAD_SIZE,
                };
                transform(data)
                  .then((response) => {
                    console.log("success");
                    console.log(response.data);
                    setAfter(response.data.output);
                    setPercentage(0.0);
                    setOpen(false);
                  })
                  .catch((response) => {
                    console.log(response);
                  });
              },
              {
                orientation: true,
                canvas: true,
                crossOrigin: "anonymous",
                maxWidth: 600,
              }
            );
          }}
          imgExtension={[".jpg", ".gif", ".png", ".gif", "jpeg"]}
          maxFileSize={5242880}
        />

        <ReactCompareImage
          aspectRatio="wider"
          leftImage={before}
          rightImage={after}
          leftImageLabel="Before"
          rightImageLabel="After"
          sliderPositionPercentage={percentage}
          sliderLineColor="black"
          leftImageCss={{ borderRadius: 10 }}
          rightImageCss={{ borderRadius: 10 }}
        />

        <Chip
          color="secondary"
          label="Download"
          className={classes.chip}
          icon={<DownloadIcon style={{ marginTop: 4 }} />}
          onClick={() => {
            triggerBase64Download(after, "styled_image");
          }}
        />
      </div>
      <Backdrop
        open={open}
        style={{ zIndex: 999 }}
        onClick={() => {
          setOpen(false);
        }}
      >
        <GridLoader size={30} margin={2} color="#e63946" />
      </Backdrop>
    </Box>
  );
}
