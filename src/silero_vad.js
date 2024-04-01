'use strict';

import { Tensor, InferenceSession, env } from "onnxruntime-web";
import modelUrl from "./../model/silero_vad.onnx?url";
// import wasmURL from "onnxruntime-web/dist/ort-wasm.wasm?url";
// import wasmSimdURL from "onnxruntime-web/ort-wasm-simd.wasm?url";
// import wasmSimdThreadURL from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
// import wasmThreadURL from "onnxruntime-web/ort-wasm-threaded.wasm?url";

env.wasm.numThreads = 1;
env.wasm.wasmPaths = "ort/";
env.wasm.simd = true;

class SileroVad {
    #session;
    #h
    #c
    #sr
    #threshold = 0.5;
    #ready = false;
    #callback = null;
    #frameQueue = [];
    #audioBuf;
    #duration = 0;
    #adtived = false;

    static IGNORE_DURATION = 1.5;
    static FRAME_SAMPLES = 512;
    static SAMPLE_RATE = 16000;
    static SpeechActivity = {NoUpdate: -1, Actived: 1, Deactived: 0};

    constructor(options = {threshold: 0.5}) {
        this.#threshold = options.threshold;
        this.#audioBuf = new Float32Array(SileroVad.FRAME_SAMPLES);
        this.#audioBuf.fillLength = 0;
    }
    reset(){
        const zeroes = Array(2 * 64).fill(0);
        this.#h = new Tensor("float32", zeroes, [2, 1, 64]);
        this.#c = new Tensor("float32", zeroes, [2, 1, 64]);
    }
    
    async init(){
        this.#session = await InferenceSession.create(
            modelUrl,
            {
              executionProviders: ["wasm"],
            }
          );
        this.#sr = new Tensor("int64", [16000]);
        this.reset();
        this.#ready = true;
    }
    pushAudioFrame(audioFrame){
        this.#frameQueue.push(audioFrame);
        this.checkAudioEvent();
    }

    checkAudioEvent() {
      if(this.#frameQueue.length > 0){
        let data = this.#frameQueue.shift();

        this.#onAudioData(data).then(() => {
          /* 
          if(time>SileroVad.IGNORE_DURATION && b!==SileroVad.SpeechActivity.NoUpdate){
            if(this.#adtived!=b){
              if(typeof(this.#callback) === "function"){
                this.#callback({time: time-(SileroVad.FRAME_SAMPLES.toFixed(2)/SileroVad.SAMPLE_RATE), actived: b});
              }
            }
            this.#adtived = b;
            
            this.checkAudioEvent();
          }*/
          this.checkAudioEvent();
        }).catch((error) => {
          console.log(error);
        });
      }
    }

    async #onAudioData(data) {
      let remains = data;
      // Loop through the remaining audio data
      while(remains !=null && remains.length > 0){
        // If there is more audio data than can fit in the buffer
        if (remains.length > this.#audioBuf.length - this.#audioBuf.fillLength) {
          // Fill the buffer with as much audio data as possible
          this.#audioBuf.set(remains.slice(0, this.#audioBuf.length - this.#audioBuf.fillLength), this.#audioBuf.fillLength);
          remains = remains.slice(this.#audioBuf.length - this.#audioBuf.fillLength);
          this.#audioBuf.fillLength = this.#audioBuf.length;
        } else {
          // Fill the buffer with the remaining audio data
          this.#audioBuf.set(remains, this.#audioBuf.fillLength);
          this.#audioBuf.fillLength += remains.length;
          remains = null;
        }
        // If the buffer is full, process the audio data
        if (this.#audioBuf.fillLength == this.#audioBuf.length) {
          const ret = await this.#process(this.#audioBuf);
          if(this.#adtived!=ret){
            if(typeof(this.#callback) === "function"){
              this.#callback({time: this.#duration, actived: ret});
            }
          }
          this.#adtived = ret;
          this.#duration += this.#audioBuf.length.toFixed(2) / SileroVad.SAMPLE_RATE;
          this.#audioBuf.fillLength = 0;
        }
      }
    }

    async #process (audioFrame) {
        if(!(audioFrame.length == 512 || audioFrame.length == 1024 || audioFrame.length == 1536)){
          throw new Error("Invalid audio frame size");
        }
        if(!this.#ready){
          return false;
        }
        const t = new Tensor("float32", audioFrame, [1, audioFrame.length]);
        const inputs = {
          input: t,
          h: this.#h,
          c: this.#c,
          sr: this.#sr,
        };

        //let start = performance.now();
        const out = await this.#session.run(inputs);
        //console.log("Inference time: " + (performance.now() - start) + "ms");
        this.#h = out.hn;
        this.#c = out.cn;
        //console.log(out.output.data[0]);
        //const [isSpeech] = out.output.data;
        //const notSpeech = 1 - isSpeech
        return out.output.data[0]>=this.#threshold;
      }
      
      /**
       * Set the callback function to be called when a speech event occurs.
       * @param {function} callback - The callback function to be called when a speech event occurs.
       */
      set onSpeechEvent(callback) {
        this.#callback = callback;
      }
};

export default SileroVad;