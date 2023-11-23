const fs = require('fs');

// 族群大小
const POPULATION_SIZE = 30;

// 迭代次數 (終止條件)
const ITERATION_NUMBER = 5000;

// 基因突變率 80%
const MUTATION_RATE = 0.8;

// 基因突變方式為多筆變更機率
const MURATION_MORE = 0.3;

let cities = [];

// 儲存下一代用來交配的親代
let gCurrentParents = {};
let gGenerationNum = 0;
let gFirstGeneration = [];
let gResult = [];

let env = { 
  x: process.argv[2].split(' ').map(item => Number(item)),
  y: process.argv[3].split(' ').map(item => Number(item)),
  index: process.argv[4].split(' ').map(item => Number(item)),
};


init();

data = gResult[0].path.map(item => item.name).join(' ')
console.log(data)


// draw(gResult[0].path);

function init() {
  cities = getMultiPoint(env.x.length);
  // 親代染色體
  gFirstGeneration = getParents(cities);
  gCurrentParents = findExcellent(gFirstGeneration);
  gResult = gFirstGeneration;
  generateChildren();
}

function random(range, offset = 0) {
  return Math.floor(Math.random() * (range - offset)) + offset;
}

function getDistance(cities) {
  let sum = 0;
  for(let i=0; i<cities.length -1; i++) {
    sum = sum + calcDistance(cities[i], cities[i + 1]);
  }
  sum = sum + calcDistance(cities[cities.length -1], cities[0]);
  return sum;
}

function calcDistance(city1, city2) {
  x = city1.x - city2.x;
  y = city1.y - city2.y;
  return Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2));
}

function findExcellent(chromosome) {
  return { chromosome1: chromosome[0], chromosome2: chromosome[1] };
}

function sort(arr) {
  return arr.sort((a,b) => a.distance - b.distance);
}

function swap(arr, index1, index2) {
	let newArr = [...arr];
	let temp;
	temp = newArr[index1];
	newArr[index1] = newArr[index2];
	newArr[index2] = temp;
	return newArr;
}

function getExcellentResult(origin, newData) {
  const excellent = newData[0];
  const originTop50 = origin.slice(0, POPULATION_SIZE - 2);
	// 保留前一代最優秀的解
  return sort(originTop50.concat(excellent).concat(excellent));
}

function getMultiPoint(num) {
  const points = [];

  for(let i = 0; i <num; i++) {
    let x = env.x[i]
    let y = env.y[i]
    let index = env.index[i]
    points.push({ name: index, x, y, dataRate: 0 });
  }
  return points;
}

// 取得親代染色體
function getParents(genes) {
  let parents = []
  
  for(let i=0; i<POPULATION_SIZE; i++) {
    let newGenes = [...genes];
    const chromosome = [];
    while(newGenes.length) {
      const index = random(newGenes.length);
      chromosome.push(newGenes[index]);
      newGenes.splice(index, 1);
    }
    const distance = getDistance(chromosome);
    parents.push({ path: chromosome, distance});
  }
  return sort(parents);
}

// 產生子代
function generateChildren() {
  let children = [];
  for(let i=0; i<ITERATION_NUMBER; i++) {
    const nextGeneration = crossover(i);
    const newGeneration = getExcellentResult(gResult, nextGeneration);
    gCurrentParents = findExcellent(newGeneration);
    gResult = newGeneration;
    children.push(newGeneration);
    // console.log(`=============== Generation ${i + 1}===============`)
    // console.log([...gResult][0].distance);
		gGenerationNum = i;
  }
  
  return children;
}

// 進行交配，產生新的基因序
function crossover(index) {
  let children = [];
  for(let i=0; i<POPULATION_SIZE; i++) {
    const child = getChild(index);
    const result = mutation(child);
    const distance = getDistance(result);
    children.push({ path: result, distance});
  }
  return sort(children);
}

function getChild(index){
  const { chromosome1, chromosome2 } = gCurrentParents;
  let chromosome = cities.map(item => '_');
  let newArr = [...chromosome2.path];

  for(let i=0; i<cities.length / 2; i++) {
    const index = random(cities.length)
    const value = chromosome1.path[index];
    chromosome[index] = value;
    newArr = newArr.filter(item => item.name !== value.name);
  }
  
  chromosome.forEach((item, index) => {
    if (item === '_') chromosome[index] = newArr.shift();
  })
  return chromosome;
}

// 基因突變
function mutation(chromosome) {
  const newChromosome = [...chromosome];
  const shouldMutation = (MUTATION_RATE * 100) >= random(100);
  return shouldMutation ? exChangeGene(newChromosome) : newChromosome;
}

// 基因突變 - 其中某兩段基因進行對調
function exChangeGene(chromosome) {
	const isMore50 = random(100) > MURATION_MORE * 100;
	return isMore50 ? swapMoreGene(chromosome) :swapOneGene(chromosome);
}

function swapMoreGene(chromosome) {
	let newChromosome = [...chromosome];
  const index1 = random(chromosome.length);
  const index2 = random(chromosome.length, index1);
	for(let i=0; i < (index2 - index1) / 2; i++) {
		newChromosome = swap(newChromosome, index1+i, index2-i);
	}
	return newChromosome;
}

function swapOneGene(chromosome) {
	let newChromosome = [...chromosome];
  let temp = {};
  const index1 = random(cities.length);
  const index2 = random(cities.length);
  temp = {...chromosome[index1]};
  newChromosome[index1] = {...chromosome[index2]};
  newChromosome[index2] = temp;
  return newChromosome;
}
