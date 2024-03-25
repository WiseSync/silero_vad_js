'use strict';

import { Tensor, InferenceSession, env } from "onnxruntime-web";
import modelUrl from "./../model/silero_vad.onnx";

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
          this.#sr = new Tensor("int64", [16000n]);
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
        const time = (this.#duration += data.length.toFixed(2) / SileroVad.SAMPLE_RATE);

        this.onAudioData(data).then((b) => { 
          if(time>SileroVad.IGNORE_DURATION && b!==SileroVad.SpeechActivity.NoUpdate){
            if(this.#adtived!=b){
              if(typeof(this.#callback) === "function"){
                this.#callback({time: time-(SileroVad.FRAME_SAMPLES.toFixed(2)/SileroVad.SAMPLE_RATE), actived: b});
              }
            }
            this.#adtived = b;
            
            this.checkAudioEvent();
          }
        }).catch((error) => {
          console.log(error);
        });
      }
    }

    async onAudioData(data) {
      let b = SileroVad.SpeechActivity.NoUpdate;
      let remains = data;
      while(remains !=null && remains.length > 0){
        if (remains.length > this.#audioBuf.length - this.#audioBuf.fillLength) {
          this.#audioBuf.set(remains.slice(0, this.#audioBuf.length - this.#audioBuf.fillLength), this.#audioBuf.fillLength);
          remains = remains.slice(this.#audioBuf.length - this.#audioBuf.fillLength);
          this.#audioBuf.fillLength = this.#audioBuf.length;
        } else {
          this.#audioBuf.set(remains, this.#audioBuf.fillLength);
          this.#audioBuf.fillLength += remains.length;
          remains = null;
        }
        if (this.#audioBuf.fillLength == this.#audioBuf.length) {
          const ret =await this.process(this.#audioBuf);
          if(ret){
            b = SileroVad.SpeechActivity.Actived;
          }else{
            if(b!==SileroVad.SpeechActivity.Actived){
              b = SileroVad.SpeechActivity.Deactived;
            }
          }
          this.#audioBuf.fillLength = 0;
        }
      }
    
      return b;
    }

    async process (audioFrame) {
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

      set onSpeechEvent(callback) {
        this.#callback = callback;
      }
};

export default SileroVad;