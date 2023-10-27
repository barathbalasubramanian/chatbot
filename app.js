

(async function() {
    async function loadModel() {
    try {
        console.log('Loading');
        const model = await tf.loadLayersModel('model/model.json');
        return model;
    } catch (error) {
        console.error("Error loading the model:", error);
    }
    }
    model = await loadModel();
})();



let wordIndex, classesList , intentsData , model , res; 
async function loadTokenizer() {
const response = await fetch('model/tokenizer.json');
wordIndex = await response.json();
}

async function loadLabelEncoder() {
const response = await fetch('model/label_encoder.json');
classesList = await response.json();
}

function tokenize(sentence) {
const words = sentence.split(' ');
return words.map(word => wordIndex[word.toLowerCase()] || 0);
}

function padSequences(sequence, maxlen) {
while (sequence.length < maxlen) {
    sequence.push(0);
}
return sequence.slice(0, maxlen);
}

async function predict(inp, maxlen) {
const tokenized = tokenize(inp);
const paddedSequence = padSequences(tokenized, maxlen);
const inputTensor = tf.tensor2d([paddedSequence]);
const prediction = await model.predict(inputTensor);    
const predictionArray = await prediction.array();
const predictedIndex = argmax(predictionArray[0]);
const predictedTag = classesList[predictedIndex];
return predictedTag;
}

function argmax(array) {
return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function getResponseForTag(tag) {
    const intent = intentsData['intents'].find(i => i['tag'] === tag);

    if (!intent) return null;

    const randomIndex = Math.floor(Math.random() * intent['responses'].length);
    res = intent['responses'][randomIndex];
    return intent['responses'][randomIndex];
}

async function runPrediction(inputSentence, maxLength) {
await loadTokenizer();
await loadLabelEncoder();
await loadIntents();

const predictedTag = await predict(inputSentence, maxLength);
const response = getResponseForTag(predictedTag);

console.log("ChatBot:", response);
}

async function loadIntents() {
const response = await fetch('model/intents.json');
intentsData = await response.json();
}

// await runPrediction(inputSentence, maxLength);

document.getElementById('form').addEventListener('submit', async function(e){
    document.querySelector('.res').innerHTML = ""
    e.preventDefault(); 
    const maxLength = 20;
    let inputSentence = e.target[0].value; 
    await runPrediction(inputSentence, maxLength);
    console.log()
    
    displaySentence(inputSentence, e.target[0]);
    
});


function displaySentence(sen , inputElement) {

    let index = 0;  
    function displayNextCharacter() {
        if (index < res.length) {
            document.querySelector('.res').innerHTML += res[index];
            index++;
            setTimeout(displayNextCharacter, 100);  
        } else {
            inputElement.value = '';  
        }
    }

    displayNextCharacter(); 
}

