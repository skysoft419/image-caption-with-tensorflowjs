/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

//import * as tf from '@tensorflow/tfjs';
import * as tf from '@tensorflow/tfjs';
import 'babel-polyfill';
var heapq = require('heapq');
//global.fetch = require('node-fetch');
//const tf = require('@tensorflow/tfjs')
// Load the binding (CPU computation)
//require('@tensorflow/tfjs-node')
//import {IMAGENET_CLASSES} from './imagenet_classes';

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    //'file:///saved_web/model.json'
    './saved_web/model.json'

const IMAGE_SIZE = 299;
const TOPK_PREDICTIONS = 10;

let model;
const imagecaptionDemo = async () => {
    status('Loading model...');

    model = await tf.loadGraphModel(MOBILENET_MODEL_PATH);

    // Warmup the model. This isn't necessary, but makes the first prediction
    // faster. Call `dispose` to release the WebGL memory allocated for the return
    // value of `predict`.
    //model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

    status('model loaded');


    document.getElementById('file-container').style.display = '';
};

class Caption {
    constructor(sentence, state, logprob, score) {
        this._sentence = sentence;
        this._state = state;
        this._logprob = logprob;
        this._score = score;
    }
    get state(){
        return this._state;
    }
    set state(state){
        this._state=state;
    }
    get sentence(){
        return this._sentence;
    }
    set sentence(sentence){
        this._sentence=sentence;
    }
}


class TopN {
    constructor() {
        this._n = 3;
        this._data = []
    }
    size() {
        return this._data.length;
    }
    pushe(newele) {
        if (this._data.length < this._n) {
            console.log('pusheeed')
            //  block of code to be executed if the condition is true
            var cmp = function(x, y) {
                if (x._score == y._score)
                    return 1;
                else if( x._score < y._score)
                    return 0;
                else
                    return 1;
            }
            heapq.push(this._data, newele, cmp);
        } else {
            //  block of code to be executed if the condition is false
            console.log("popped")
            heapq.pushpop(this._data, newele, cmp);
        }
    }
    extract(){
        var data = [];
        data = this._data;
        this._data = null
        return data;

    }
    reset(){
        this._data = []
    }
}



//var my = new Array();
//my.push(new car("Ford", "Black", 2017, 15000));
//my.push(new car("Hyundai", "Red", 2017, 17000));


/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
    status('Predicting...');

    // The first start time includes the time it takes to extract the image
    // from the HTML and preprocess it, in additon to the predict() call.
    const startTime1 = performance.now();
    // The second start time excludes the extraction and preprocessing and
    // includes only the predict() call.
    let startTime2;
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const inputTensor = tf.browser.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(0.5);
    let resized = tf.image.resizeBilinear(inputTensor,[299, 299],false);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = resized.div(tf.scalar(255.0)).sub(offset).mul(tf.scalar(2.0));
    //const normalized = resized.div(tf.scalar(255.0)).sub(offset);
    let img = normalized.expandDims()
    // Reshape to a single-element batch so we can pass it to predict.
    //const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    let start = (new Date()).getTime()
    // https://js.tensorflow.org/api/latest/#tf.Model.predict

    //_____________________________________________________________________________________

    var partial_caption_beam = new TopN()
    var complete_captions = new TopN()

    var feed_image = model.execute({'ExpandDims_4': img.asType('float32')},"lstm/initial_state");
    var initial_state = feed_image.squeeze().dataSync();
    var init_sentence = [];
    init_sentence.push(2);
    console.log(img.arraySync())

    var initial_beam = new Caption(init_sentence,initial_state,0.0,0.0);
    console.log(initial_beam._state);
    console.log(initial_beam._logprob);
    partial_caption_beam.pushe(initial_beam);

    var k;

    let partial_caption_list = [];
    tf.tidy(() => {
        for(k=0; k < 20 ; k++){
            console.log('++++++++++++++++++++ itreation ++++++++++++++++++++++++++++++')
            console.log(k)
            partial_caption_list = partial_caption_beam.extract();
            console.log(partial_caption_list);
            partial_caption_beam.reset();
            var input_feed=[];
            let state_feed=[];

            partial_caption_list.forEach((element, index, array) => {
                input_feed.push(element._sentence[element._sentence.length-1]);
                let a = partial_caption_list[index]._state;
                state_feed.push(a);

            });
            //const state_feed = tf.stack([partial_caption_list[0]._state,partial_caption_list[1]._state]);
            console.log('feed')
            console.log(tf.tensor(state_feed))
            console.log(input_feed)
            const new_states_op = model.execute({'ExpandDims_4': img.asType('float32'),'lstm/state_feed':tf.tensor(state_feed),'input_feed':tf.tensor(input_feed)},"lstm/state");
            const softmax_op = model.execute({'ExpandDims_4': img.asType('float32'),'lstm/state_feed':tf.tensor(state_feed),'input_feed':tf.tensor(input_feed)},"softmax:0");
//		const softmax=softmax_op.dataSync();
//		const new_states=new_states_op.dataSync();
            for (let [index, partial_caption] of partial_caption_list.entries()){

                var word_probabilities = [];
                var state;

                console.log(partial_caption)
                word_probabilities.push(softmax_op.slice([index],[1,12000]).squeeze().arraySync());
                //console.log(word_probabilities)

                state = new_states_op.slice([index],[1,1024]).squeeze().arraySync();

                var tuples=[];
                var words_and_probs_enum = word_probabilities[0].entries();
                for (let [x,y] of words_and_probs_enum){
                    tuples.push([x,y]);
                }
                function sortFloat(a, b) {
                    return (Math.abs(a[1]) * -1)-(Math.abs(b[1]) * -1);
                }
                let sorted_tup=tuples.sort(sortFloat);
                var logprob;
                var score;
                for (var j = 0; j < 3; j++) {
                    var sen = [];
                    var w = tuples[j][0];
                    var p = tuples[j][1];
                    console.log(w);
                    console.log(p);
                    if (p < 0.000000000001) {
                        continue;
                    }
                    sen = partial_caption._sentence.concat(w);
                    //sen.push(w);
                    logprob = partial_caption._logprob + Math.log(p);
                    score = logprob;
                    if(w == 1) {
                        let beam = new Caption(sen, state, logprob, score);
                        complete_captions.pushe(beam);
                    } else {
                        let beam = new Caption(sen, state, logprob, score);
                        partial_caption_beam.pushe(beam);
                    }
                }
                if (partial_caption_beam.size() == 0){
                    console.log('WE RUN OUT OF PARTIAL CAPTIONS HAPPENS WHEN BEAM SIZE = 1')
                    break;
                }
            }
            partial_caption_list=[]
            //if (complete_captions.size() == 0){
            //	console.log("should not happen")
            //	complete_captions = partial_caption_beam
            //}
        }
    });
    let final_c = complete_captions.extract();
    console.log(final_c);
    var fs = require("fs");
    var text = fs.readFileSync("./dist/word_counts.txt","utf-8");
    var textByLine = text.split("\n")
    var caption_1 = []
    for (let [index, final_caption] of final_c.entries()){
        console.log(final_caption._sentence)
        for(let v=0;v<final_caption._sentence.length;v++){
            if (v == 0 || v == (final_caption._sentence.length-1)){
                continue
            }
            else{
                var tid = final_caption._sentence[v]
                var text_get_token = textByLine[tid].split(" ")
                console.log(text_get_token[0])
                caption_1.push(text_get_token[0])
            }
        }

        break
    }
    var caption_to_html = caption_1.toString().replace( /,/g, " " );
    showResults(imgElement, caption_to_html);

    let end = (new Date()).getTime()

//    await processOutput(output)

    //message(`inference ran in ${(end - start) / 1000} secs`, true)
    //enableElements()
    console.log('done')
    startTime2 = performance.now();
    //});

}

//
// UI
//

function showResults(imgElement, caption_to_html) {
    const predictionContainer = document.createElement('div');
    predictionContainer.className = 'pred-container';

    const imgContainer = document.createElement('div');
    imgContainer.innerHTML = "";
    predictionContainer.innerHTML = "";
    predictionsElement.innerHTML = "";
    imgContainer.appendChild(imgElement);
    //imgContainer.replaceChild(imgElement,imgContainer.childNodes[0]);
//  predictionContainer.replaceChild(imgContainer,imgContainer.childNodes[0]);
    predictionContainer.appendChild(imgContainer);
    document.getElementById("predicted_caption").innerHTML = caption_to_html;

    predictionsElement.insertBefore(
        predictionContainer, predictionsElement.firstChild);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
    let files = evt.target.files;
    // Display thumbnails & issue call to predict each image.
    for (let i = 0, f; f = files[i]; i++) {
        // Only process image files (skip non image files)
        if (!f.type.match('image.*')) {
            continue;
        }
        let reader = new FileReader();
        const idx = i;
        // Closure to capture the file information.
        reader.onload = e => {
            // Fill the image & call predict.
            let img = document.createElement('img');
            img.src = e.target.result;
            img.width = IMAGE_SIZE;
            img.height = IMAGE_SIZE;
            img.onload = () => predict(img);
        };

        // Read in the image file as a data URL.
        reader.readAsDataURL(f);
    }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

imagecaptionDemo();
