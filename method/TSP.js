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
  // x: [744,458,263,684,708,533,502,927,635,598,854,814,617,558,1465,676,693,379,456,1546,340,319,447,1396,601,669,657,442,557,540,383,724,529,635,601,573,1617,565,1022,606,796,827,1244,1075,599,392,618,1645,567,452,766,404,1251,1430.7310043278455,1288,663,620,821,465,533,368,899,306,1288,1369,526,1362,584,574,965,1180,1519,856,1211,905,623,395,564,350,190,1339,1134,802,220,486,963,709,642,1342,809,1198,563,469,861,590,436,459,675,313,1128,541,531,805,1340,1316,1256,546,1571,814,430,1043,800,553,628,735,814,1562,928,1857,690,884,1108,551,1416,929,944,334,738,705,1194,687,581,1377,1010,614,661,661,840,1173,1093,633,992,467,668,1123,844,762,1195,1013,467,1056,551,824,758,1095,889,774,903,789,487,411,693,890,1351,642,865,851,1300,1147,678,570,1320,603,1232,391,851,752,1077,972,858,1219,1174,431,845,1181.1964569868358,844,424,652,468,988,718,640,943,769,556,999,625,924,1127,1505,1376,1314,667,1125,312,1345,1169,1139,624,1321,1368,1233,1148,1457,1169,1175,493,1152,371,1094,461,1654,619,1525,1400,1308,901,1092,1289,1167,820,1072,462,1145,1409,1592,1088,1596,477,1244,885,1269,1171,1043,1547,1553,920,1455,691,1274,566,1296,1364,1387,707,1104,1226,1210,992,1451,1532,993,1138,1376,1237,992,856,1393,503,1198,1030,1214,1487,1611,1194,1103,1648,1216,716,1536,746,1188,1701,1330,1225,1605,995,1085,506,1366,727,1529,1277,1211,1177,1399,1046,1277,680,1170,1209,1145,1037.3405528456092,1488,1349,1348,605,1541,1010,1424,1453,1356,509,1670,587,1190,1380,1456,1280,1489,1410,1461,1322,1270,764,1681,521,1540,778,1652,1156,1631,1234,1489,1350,1363,1148,1333,762,1340,478,1104,1390,1207,1214,1554,1398,1515,431,1357,1383,1776,529,1344,1455,1719,1566,1157,1551,1280,647,1535,1526,1640,1422.329157671894,1451,1140,1300,1232,1297,888,1402,680,1172,1420.2543402769656,1275,371,1190,909,1606,554,1618,437,1538,1247,1086,1190,1352,1737,1319,395,434.62788990775493,636,1863.4232636541415,1046,821.4477948079533,378,502.8929725516045,540,1413.276650046229],
  // y: [691,465,404,605,608,556,643,330,1432,1002,512,1147,397,297,829,711,850,453,281,1456,434,300,1622,1091,700,462,1147,1563,633,1516,961,967,534,1067,1519,1289,1318,659,270,1549,1346,1392,1240,1249,338,271,922,1159,858,955,1317,377,1119,500.5096981916506,487,1444,510,690,1517,432,555,699,980,454,935,858,1059,276,623,627,791,1488,1093,1512,1399,890,501,1120,400,415,1494,288,1506,260,1319,305,552,1583,543,1646,607,406,477,702,325,1321,321,1250,537,628,268,1289,395,814,675,1504,1489,731,1281,1318,331,1460,718,1279,1338,1287,1466,1232,1154,1250,1546,1510,1192,677,1606,498,367,1668,1058,752,930,1211,376,1231,1400,967,475,1059,1300,1334,1498,335,876,1369,1095,1037,479,992,1627,1086,1268,1532,1312,1285,669,1241,800,1125,314,1325,1509,1573,588,685,1506,1635,824,1401,575,1324,935,1130,403,1545,1395,1393,654,1479,1223,1768,315,1576,710,1326,449.4440829022728,1409,631,1513,719,1309,1401,1132,796,1539,364,357,1541,442,495,1431,1622,514,1057,1211,298,753,991,692,1531,733,419,347,588,522,1121,320,1316,285,372,449,220,1148,446,671,541,703,1316,692,251,604,1500,770,509,612,1390,677,460,406,1543,637,1299,858,643,380,800,1351,535,1304,635,657,442,808,851,878,1615,1184,356,344,541,601,1320,650,621,335,421,835,1219,788,1475,720,1116,675,1383,1425,255,581,1425,1321,533,789,1064,909,1453,633,572,830,886,1010,434,221,1256,777,689,1060,1501,1501,471,1176,1364,1141,1360,1241,783.9604715206656,1474,897,1151,1136,1379,1249,968,461,1109,1223,1395,1075,1168,775,1258,744,1171,1363,1359,640,1079,792,1178,970,1323,1488,1235,481,1614,697,1426,845,1183,864,1370,481,1367,357,1332,1360,1307,1094,1567,998,1211,1376,1529,1400,1144,1279,1102,1169,1076,701,1297,465,1533,942,1158,720,1205,159.7775333691369,1254,1166,893,760,907,1689,1198,780,1493,1516.5337993175256,1242,526,1415,1105,1126,470,1270,1002,1155,397,1503,1097,1170,1284,1289,972,1025.5880266551144,1233,1089.1879897453127,1203,334.7947987533519,320,800.2703662699346,801,1050.269728170366],
};

// console.log(env.x, env.y)



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
