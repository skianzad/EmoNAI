/** PowerPoint-style stamp library. Bodies are 64×64 SVG children. */

const S = {
  fill: "fill:none;stroke:#141414;stroke-width:2.3;stroke-linecap:round;stroke-linejoin:round",
  shape: "fill:rgba(20,20,20,0.14);stroke:#141414;stroke-width:2.2;stroke-linejoin:round",
  solid: "fill:#141414;stroke:#141414;stroke-width:1.4;stroke-linejoin:round",
};

function item(id, en, zh, aliases, body, kind = "icon") {
  return { id, en, zh, aliases, body, kind };
}

export const SYMBOLS = {
  dog: item("dog", "Dog", "狗", ["dog", "dogs", "puppy", "puppies", "hound", "狗", "犬"], `<g style="${S.fill}"><ellipse cx="40" cy="38" rx="15" ry="11"/><circle cx="24" cy="30" r="9"/><path d="M18 28 L6 32 L18 38 Z"/><path d="M26 22 Q18 10 14 30"/><circle cx="20" cy="28" r="1.6" fill="#141414"/><path d="M10 36 Q14 44 8 40"/><path d="M30 32 Q34 42 28 44"/><path d="M30 48 v10 M38 49 v10 M46 49 v10 M54 48 v8"/><path d="M54 32 Q64 14 58 40"/></g>`),
  cat: item("cat", "Cat", "猫", ["cat", "cats", "kitten", "kittens", "kitty", "猫"], `<g style="${S.fill}"><path d="M22 54 Q16 36 24 26 Q32 20 40 26 Q48 36 42 54 Z"/><circle cx="32" cy="20" r="10"/><path d="M24 14 L20 4 L30 12"/><path d="M40 14 L44 4 L34 12"/><path d="M27 20 L30 20 M34 20 L37 20"/><path d="M32 22 L32 25"/><path d="M22 24 L12 20 M22 26 L12 28 M42 24 L52 20 M42 26 L52 28"/><path d="M42 46 Q58 42 54 24"/></g>`),
  bird: item("bird", "Bird", "鸟", ["bird", "birds", "sparrow", "dove", "pigeon", "eagle", "鸟", "雀"], `<g style="${S.fill}"><ellipse cx="34" cy="34" rx="14" ry="8"/><circle cx="48" cy="30" r="5"/><path d="M52 30 L60 28 L52 33"/><path d="M28 32 Q18 18 8 28 Q22 30 28 36"/><path d="M30 42 L26 52M36 42 L38 52"/><path d="M48 26 Q50 20 46 20"/></g>`),
  fish: item("fish", "Fish", "鱼", ["fish", "fishes", "goldfish", "鱼"], `<g style="${S.fill}"><ellipse cx="30" cy="32" rx="16" ry="9"/><path d="M46 32 L60 22 L56 32 L60 42 Z"/><circle cx="20" cy="30" r="1.4" fill="#141414"/><path d="M28 24 Q32 32 28 40"/></g>`),
  horse: item("horse", "Horse", "马", ["horse", "horses", "pony", "马"], `<g style="${S.fill}"><ellipse cx="36" cy="38" rx="18" ry="9"/><path d="M20 34 Q16 22 12 14 Q10 10 14 10 Q20 12 22 22"/><path d="M16 14 Q22 8 20 20"/><circle cx="12" cy="12" r="1.2" fill="#141414"/><path d="M54 36 Q62 48 52 50"/><path d="M24 46 L22 58 M32 47 L32 58 M42 47 L44 58 M50 46 L52 56"/></g>`),
  rabbit: item("rabbit", "Rabbit", "兔", ["rabbit", "rabbits", "bunny", "bunnies", "兔", "兔子"], `<g style="${S.fill}"><ellipse cx="34" cy="42" rx="12" ry="9"/><circle cx="28" cy="30" r="7"/><path d="M24 24 L22 8 L27 24"/><path d="M30 24 L34 8 L34 24"/><path d="M44 40 Q52 32 48 46"/><circle cx="26" cy="29" r="1" fill="#141414"/></g>`),
  butterfly: item("butterfly", "Butterfly", "蝴蝶", ["butterfly", "butterflies", "蝴蝶"], `<g style="${S.fill}"><path d="M32 16 L32 50"/><path d="M32 28 Q12 10 10 28 Q14 40 32 34"/><path d="M32 28 Q52 10 54 28 Q50 40 32 34"/><path d="M32 36 Q16 44 18 52 Q26 50 32 42"/><path d="M32 36 Q48 44 46 52 Q38 50 32 42"/></g>`),
  person: item("person", "Person", "人", ["person", "people", "human", "someone", "人"], `<g style="${S.fill}"><circle cx="32" cy="12" r="7"/><path d="M32 19 v6"/><path d="M20 28 q12-8 24 0 v18 q-12 6-24 0 z"/><path d="M22 27 q-8 6-6 16"/><path d="M42 27 q8 6 6 16"/><path d="M24 46 v14 M28 46 v14"/><path d="M36 46 v14 M40 46 v14"/><path d="M22 60 h8 M34 60 h8"/></g>`),
  man: item("man", "Man", "男人", ["man", "men", "guy", "male", "gentleman", "3d man", "男人", "男"], `<g style="${S.fill}"><ellipse cx="32" cy="11" rx="7" ry="7.5"/><path d="M27 8 q5-4 10 0"/><path d="M29 12 h1.5 M35 12 h1.5"/><path d="M30 16 q2 1.5 4 0"/><path d="M32 18.5 v4"/><path d="M24 24 q8-5 16 0 l3 20 q-11 5-22 0 z"/><path d="M26 28 h12"/><path d="M22 26 q-7 8-5 18"/><path d="M42 26 q7 8 5 18"/><path d="M26 44 l-1 16 M30 45 l1 16"/><path d="M34 45 l-1 16 M38 44 l1 16"/><path d="M23 60 h8 M33 60 h8"/></g>`),
  woman: item("woman", "Woman", "女人", ["woman", "women", "lady", "female", "girl", "3d woman", "女人", "女"], `<g style="${S.fill}"><path d="M22 12 q4-8 10-8 q6 0 10 8 q1 6-4 8 q-6 2-12 0 q-5-2-4-8 z"/><circle cx="32" cy="12" r="6.5"/><path d="M29 12 h1.2 M35 12 h1.2"/><path d="M30 16 q2 1.4 4 0"/><path d="M32 18.5 v4"/><path d="M24 24 q8-5 16 0 l8 32 q-16 6-32 0 z"/><path d="M22 26 q-6 7-4 16"/><path d="M42 26 q6 7 4 16"/><path d="M28 54 v8 M36 54 v8"/><path d="M26 62 h6 M32 62 h6"/></g>`),
  child: item("child", "Child", "小孩", ["child", "children", "kid", "kids", "baby", "小孩", "孩子"], `<g style="${S.fill}"><circle cx="32" cy="14" r="7"/><path d="M32 21 v4"/><path d="M22 28 q10-6 20 0 v14 q-10 5-20 0 z"/><path d="M24 27 q-6 5-4 12"/><path d="M40 27 q6 5 4 12"/><path d="M26 42 v14 M30 42 v14"/><path d="M34 42 v14 M38 42 v14"/><path d="M24 56 h8 M32 56 h8"/></g>`),
  tree: item("tree", "Tree", "树", ["tree", "trees", "pine", "oak", "树", "树木"], `<g style="${S.fill}"><path d="M32 58 L32 40"/><path d="M32 18 L16 40 L48 40 Z"/><path d="M32 8 L20 28 L44 28 Z"/></g>`),
  flower: item("flower", "Flower", "花", ["flower", "flowers", "rose", "daisy", "花"], `<g style="${S.fill}"><circle cx="32" cy="24" r="5"/><circle cx="32" cy="12" r="6"/><circle cx="22" cy="18" r="6"/><circle cx="42" cy="18" r="6"/><circle cx="24" cy="30" r="6"/><circle cx="40" cy="30" r="6"/><path d="M32 30 L32 56"/><path d="M32 44 Q22 40 18 48"/></g>`),
  sun: item("sun", "Sun", "太阳", ["sun", "sunny", "sunshine", "太阳", "日"], `<g style="${S.fill}"><circle cx="32" cy="32" r="10"/><path d="M32 8 L32 14M32 50 L32 56M8 32 L14 32M50 32 L56 32M14 14 L18 18M46 46 L50 50M50 14 L46 18M14 50 L18 46"/></g>`),
  moon: item("moon", "Moon", "月亮", ["moon", "crescent", "月亮", "月"], `<g style="${S.fill}"><path d="M40 12 A16 16 0 1 0 40 52 A12 12 0 1 1 40 12 Z"/></g>`),
  cloud: item("cloud", "Cloud", "云", ["cloud", "clouds", "cloudy", "云"], `<g style="${S.fill}"><path d="M18 40 H46 A10 10 0 0 0 46 20 A14 14 0 0 0 20 24 A10 10 0 0 0 18 40 Z"/></g>`),
  mountain: item("mountain", "Mountain", "山", ["mountain", "mountains", "hill", "hills", "山", "山峰"], `<g style="${S.fill}"><path d="M4 52 L22 18 L36 40 L44 28 L60 52 Z"/><path d="M16 28 L22 18 L28 28"/></g>`),
  rain: item("rain", "Rain", "雨", ["rain", "rainy", "雨"], `<g style="${S.fill}"><path d="M18 28 H46 A9 9 0 0 0 46 12 A12 12 0 0 0 22 16 A8 8 0 0 0 18 28 Z"/><path d="M22 36 L18 48M32 36 L28 50M42 36 L38 48"/></g>`),
  car: item("car", "Car", "车", ["car", "cars", "auto", "vehicle", "车", "汽车"], `<g style="${S.fill}"><path d="M10 36 L14 26 L24 20 L42 20 L54 28 L58 36 Z"/><path d="M8 36 H56 V44 H8 Z"/><circle cx="20" cy="44" r="5"/><circle cx="46" cy="44" r="5"/></g>`),
  bus: item("bus", "Bus", "公交", ["bus", "buses", "公交", "巴士"], `<g style="${S.fill}"><rect x="8" y="16" width="48" height="28" rx="4"/><path d="M8 30 H56"/><rect x="14" y="20" width="10" height="8"/><rect x="28" y="20" width="10" height="8"/><rect x="42" y="20" width="8" height="8"/><circle cx="20" cy="46" r="5"/><circle cx="46" cy="46" r="5"/></g>`),
  plane: item("plane", "Plane", "飞机", ["plane", "planes", "airplane", "aircraft", "jet", "飞机"], `<g style="${S.fill}"><path d="M8 34 L56 28 L58 32 L8 40 Z"/><path d="M28 30 L18 16 L24 16 L36 30"/><path d="M28 36 L20 50 L26 50 L36 36"/><path d="M50 30 L58 18 L54 32"/></g>`),
  bike: item("bike", "Bike", "自行车", ["bike", "bikes", "bicycle", "cycle", "自行车"], `<g style="${S.fill}"><circle cx="16" cy="42" r="10"/><circle cx="48" cy="42" r="10"/><path d="M16 42 L28 24 L44 24 L48 42 M28 24 L24 42 M28 24 L36 16"/></g>`),
  boat: item("boat", "Boat", "船", ["boat", "boats", "ship", "sailboat", "船"], `<g style="${S.fill}"><path d="M8 40 L16 50 H48 L56 40 Z"/><path d="M32 40 L32 12 L50 40"/></g>`),
  house: item("house", "House", "房子", ["house", "home", "building", "房子", "家"], `<g style="${S.fill}"><path d="M8 32 L32 12 L56 32"/><rect x="14" y="32" width="36" height="22"/><rect x="26" y="40" width="12" height="14"/></g>`),
  cup: item("cup", "Cup", "杯子", ["cup", "mug", "coffee", "tea", "杯子", "咖啡"], `<g style="${S.fill}"><path d="M16 18 H42 V40 A10 10 0 0 1 16 40 Z"/><path d="M42 22 H50 A6 6 0 0 1 50 34 H42"/><path d="M18 50 H40"/></g>`),
  phone: item("phone", "Phone", "手机", ["phone", "phones", "iphone", "mobile", "手机"], `<g style="${S.fill}"><rect x="20" y="8" width="24" height="48" rx="4"/><circle cx="32" cy="50" r="2"/><path d="M28 12 H36"/></g>`),
  lamp: item("lamp", "Lamp", "灯", ["lamp", "light", "灯"], `<g style="${S.fill}"><path d="M20 28 H44 L38 12 H26 Z"/><path d="M32 28 V46"/><path d="M22 50 H42"/><circle cx="32" cy="8" r="3"/></g>`),
  book: item("book", "Book", "书", ["book", "books", "notebook", "书"], `<g style="${S.fill}"><path d="M14 12 H32 V52 H14 A4 4 0 0 1 10 48 V16 A4 4 0 0 1 14 12 Z"/><path d="M32 12 H50 A4 4 0 0 1 54 16 V48 A4 4 0 0 1 50 52 H32"/></g>`),
  chair: item("chair", "Chair", "椅子", ["chair", "chairs", "seat", "椅子"], `<g style="${S.fill}"><rect x="18" y="10" width="28" height="20"/><path d="M18 30 H46 V38 H18 Z"/><path d="M20 38 L18 54M44 38 L46 54M22 38 L22 48M42 38 L42 48"/></g>`),
  table: item("table", "Table", "桌子", ["table", "desk", "桌子", "餐桌"], `<g style="${S.fill}"><path d="M8 28 H56 V34 H8 Z"/><path d="M14 34 L12 54M50 34 L52 54M24 34 V46M40 34 V46"/></g>`),
  camera: item("camera", "Camera", "相机", ["camera", "cameras", "相机"], `<g style="${S.fill}"><rect x="8" y="22" width="48" height="28" rx="4"/><circle cx="32" cy="36" r="10"/><circle cx="32" cy="36" r="5"/><path d="M22 22 L26 14 H38 L42 22"/></g>`),
  bag: item("bag", "Bag", "包", ["bag", "bags", "handbag", "purse", "包", "手提包"], `<g style="${S.fill}"><rect x="14" y="24" width="36" height="28" rx="4"/><path d="M24 24 V18 A8 8 0 0 1 40 18 V24"/></g>`),
  clock: item("clock", "Clock", "钟", ["clock", "watch", "time", "钟", "时钟"], `<g style="${S.fill}"><circle cx="32" cy="32" r="18"/><path d="M32 32 L32 20"/><path d="M32 32 L42 36"/><circle cx="32" cy="32" r="2" fill="#141414"/></g>`),
  flag: item("flag", "Flag", "旗", ["flag", "flags", "旗"], `<g style="${S.fill}"><path d="M16 8 V56"/><path d="M16 10 H50 L42 22 L50 34 H16 Z"/></g>`),
  pin: item("pin", "Pin", "图钉", ["pin", "pins", "location", "marker", "图钉", "地点"], `<g style="${S.fill}"><path d="M32 8 A12 12 0 0 1 32 32 L32 56"/><circle cx="32" cy="20" r="5"/></g>`),
  bulb: item("bulb", "Bulb", "灯泡", ["bulb", "lightbulb", "idea", "灯泡"], `<g style="${S.fill}"><circle cx="32" cy="24" r="14"/><path d="M24 36 H40 V42 H24 Z"/><path d="M26 42 H38 V48 H26 Z"/><path d="M26 28 H38"/></g>`),
  speech: item("speech", "Speech", "对话", ["speech", "chat", "talk", "bubble", "对话"], `<g style="${S.fill}"><rect x="8" y="12" width="48" height="28" rx="8"/><path d="M20 40 L16 52 L28 40"/></g>`),
  heart: item("heart", "Heart", "心", ["heart", "love", "心", "爱心"], `<g style="${S.solid}"><path d="M32 52 L12 30 A11 11 0 0 1 32 18 A11 11 0 0 1 52 30 Z"/></g>`, "shape"),
  star: item("star", "Star", "星", ["star", "stars", "星", "星星"], `<g style="${S.solid}"><path d="M32 8 L38 24 L56 24 L42 36 L48 52 L32 42 L16 52 L22 36 L8 24 L26 24 Z"/></g>`, "shape"),
  rect: item("rect", "Rectangle", "矩形", ["rectangle", "rect", "box", "square", "矩形", "方块"], `<rect style="${S.shape}" x="10" y="16" width="44" height="32" rx="2"/>`, "shape"),
  roundrect: item("roundrect", "Round rect", "圆角矩形", ["roundrect", "rounded"], `<rect style="${S.shape}" x="8" y="16" width="48" height="32" rx="10"/>`, "shape"),
  oval: item("oval", "Oval", "椭圆", ["oval", "ellipse", "椭圆"], `<ellipse style="${S.shape}" cx="32" cy="32" rx="22" ry="14"/>`, "shape"),
  circle: item("circle", "Circle", "圆", ["circle", "circles", "圆", "圆形"], `<circle style="${S.shape}" cx="32" cy="32" r="18"/>`, "shape"),
  triangle: item("triangle", "Triangle", "三角", ["triangle", "triangles", "三角"], `<path style="${S.shape}" d="M32 10 L54 50 H10 Z"/>`, "shape"),
  diamond: item("diamond", "Diamond", "菱形", ["diamond", "rhombus", "菱形"], `<path style="${S.shape}" d="M32 8 L56 32 L32 56 L8 32 Z"/>`, "shape"),
  pentagon: item("pentagon", "Pentagon", "五边形", ["pentagon", "五边形"], `<path style="${S.shape}" d="M32 8 L54 24 L46 50 H18 L10 24 Z"/>`, "shape"),
  hexagon: item("hexagon", "Hexagon", "六边形", ["hexagon", "六边形"], `<path style="${S.shape}" d="M20 12 H44 L56 32 L44 52 H20 L8 32 Z"/>`, "shape"),
  chevron: item("chevron", "Chevron", "V形", ["chevron"], `<path style="${S.shape}" d="M10 16 H36 L54 32 L36 48 H10 L28 32 Z"/>`, "shape"),
  trapezoid: item("trapezoid", "Trapezoid", "梯形", ["trapezoid", "梯形"], `<path style="${S.shape}" d="M16 16 H48 L56 48 H8 Z"/>`, "shape"),
  plus: item("plus", "Plus", "加号", ["plus", "add", "加号"], `<path style="${S.solid}" d="M28 12 H36 V28 H52 V36 H36 V52 H28 V36 H12 V28 H28 Z"/>`, "shape"),
  minus: item("minus", "Minus", "减号", ["minus", "subtract", "减号"], `<rect style="${S.solid}" x="12" y="28" width="40" height="8" rx="2"/>`, "shape"),
  check: item("check", "Check", "对勾", ["check", "checkmark", "tick", "对勾"], `<path style="${S.fill}" d="M12 34 L26 48 L52 16"/>`, "shape"),
  cross: item("cross", "Cross", "叉", ["cross", "close", "叉"], `<path style="${S.fill}" d="M16 16 L48 48 M48 16 L16 48"/>`, "shape"),
  arrowRight: item("arrowRight", "Arrow", "箭头", ["arrow", "arrows", "rightarrow", "箭头"], `<g style="${S.solid}"><path d="M8 28 H36 V20 L56 32 L36 44 V36 H8 Z"/></g>`, "shape"),
  arrowLeft: item("arrowLeft", "Arrow left", "左箭头", ["leftarrow"], `<g style="${S.solid}"><path d="M56 28 H28 V20 L8 32 L28 44 V36 H56 Z"/></g>`, "shape"),
  arrowUp: item("arrowUp", "Arrow up", "上箭头", ["uparrow"], `<g style="${S.solid}"><path d="M28 56 V28 H20 L32 8 L44 28 H36 V56 Z"/></g>`, "shape"),
  arrowDown: item("arrowDown", "Arrow down", "下箭头", ["downarrow"], `<g style="${S.solid}"><path d="M28 8 V36 H20 L32 56 L44 36 H36 V8 Z"/></g>`, "shape"),
  arrowBoth: item("arrowBoth", "Double arrow", "双箭头", ["doublearrow", "leftright"], `<g style="${S.solid}"><path d="M8 32 L20 20 V28 H44 V20 L56 32 L44 44 V36 H20 V44 Z"/></g>`, "shape"),
  callout: item("callout", "Callout", "标注", ["callout", "calloutbox", "标注"], `<g style="${S.shape}"><rect x="8" y="10" width="48" height="28" rx="4"/><path d="M20 38 L16 52 L30 38"/></g>`, "shape"),
};

export const CATEGORIES = [
  { id: "animals", en: "Animals", zh: "动物", ids: ["dog", "cat", "bird", "fish", "horse", "rabbit", "butterfly"] },
  { id: "people", en: "People", zh: "人物", ids: ["person", "man", "woman", "child"] },
  { id: "saved", en: "Saved", zh: "已保存", ids: [] },
  { id: "nature", en: "Nature", zh: "自然", ids: ["tree", "flower", "sun", "moon", "cloud", "mountain", "rain"] },
  { id: "travel", en: "Travel", zh: "出行", ids: ["car", "bus", "plane", "bike", "boat", "house"] },
  { id: "objects", en: "Objects", zh: "物品", ids: ["cup", "phone", "lamp", "book", "chair", "table", "camera", "bag", "clock", "flag", "pin", "bulb", "speech"] },
  { id: "shapes", en: "Shapes", zh: "形状", ids: ["rect", "roundrect", "oval", "circle", "triangle", "diamond", "pentagon", "hexagon", "chevron", "trapezoid", "plus", "minus", "check", "cross", "heart", "star", "callout"] },
  { id: "arrows", en: "Arrows", zh: "箭头", ids: ["arrowRight", "arrowLeft", "arrowUp", "arrowDown", "arrowBoth"] },
];

const STOP = new Set(
  "a an the to of in on at and or with from this that add put make sketch photo scene edit a few some more please 一个 一张 添加 放 把 和 的 在".split(/\s+/)
);

const UNIT = new Set("x × * 只 条 个 位 辆 朵 棵 座 本 杯 艘 台 人".split(/\s+/));

const NUM = {
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
  一: 1,
  两: 2,
  二: 2,
  三: 3,
  四: 4,
  五: 5,
  六: 6,
  七: 7,
  八: 8,
  九: 9,
  十: 10,
};

function parseCount(word) {
  if (!word) return 0;
  if (NUM[word]) return NUM[word];
  if (/^\d+$/.test(word)) return Math.min(12, Math.max(1, Number(word)));
  return 0;
}

function quantityAround(words, i) {
  const prev = words[i - 1];
  const prev2 = words[i - 2];
  const next = words[i + 1];
  const next2 = words[i + 2];
  return (
    parseCount(prev) ||
    (UNIT.has(prev) && parseCount(prev2)) ||
    parseCount(next) ||
    (UNIT.has(next) && parseCount(next2)) ||
    1
  );
}

const ALIAS = [];
const BUILTIN_IDS = new Set(Object.keys(SYMBOLS));
const STORE = "sd-symbol-lib";

function rebuildAliases() {
  ALIAS.length = 0;
  for (const symbol of Object.values(SYMBOLS)) {
    for (const alias of symbol.aliases) {
      ALIAS.push({ alias: String(alias).toLowerCase(), id: symbol.id });
    }
  }
  ALIAS.sort((a, b) => b.alias.length - a.alias.length);
}

rebuildAliases();

const REJECTED = new Set();
const REJECTED_GONE = new Set();
const REJECTED_COUNT = new Map();

export function rejectSymbol(id, count = 0) {
  const key = String(id || "").trim();
  if (!key) return;
  REJECTED.add(key);
  REJECTED_GONE.delete(key);
  REJECTED_COUNT.set(key, Number(count) || 0);
}

export function clearRejected(id) {
  if (id) {
    const key = String(id);
    REJECTED.delete(key);
    REJECTED_GONE.delete(key);
    REJECTED_COUNT.delete(key);
    return;
  }
  REJECTED.clear();
  REJECTED_GONE.clear();
  REJECTED_COUNT.clear();
}

export function isRejected(id) {
  return REJECTED.has(String(id || ""));
}

export function syncRejected(text) {
  const wanted = Object.fromEntries(matchPrompt(text).map((item) => [item.id, item.count]));
  for (const id of [...REJECTED]) {
    const n = wanted[id] || 0;
    if (!n) {
      REJECTED_GONE.add(id);
      REJECTED_COUNT.set(id, 0);
      continue;
    }
    if (REJECTED_GONE.has(id) || n > (REJECTED_COUNT.get(id) || 0)) clearRejected(id);
  }
}

export function symbolSlug(name) {
  return (
    String(name || "")
      .toLowerCase()
      .replace(/^obj[-_]/, "")
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 32) || "symbol"
  );
}

function persistCustom() {
  if (typeof localStorage === "undefined") return;
  const customs = Object.values(SYMBOLS).filter((symbol) => symbol.custom);
  localStorage.setItem(STORE, JSON.stringify(customs));
}

export function addCustomSymbol(raw, { persist = true } = {}) {
  const en = String(raw.en || raw.id || "symbol").trim() || "symbol";
  const id = symbolSlug(raw.id || en);
  if (!raw.body) return SYMBOLS[id];
  const aliases = [...new Set([id, en.toLowerCase(), ...(raw.aliases || []).map((item) => String(item).toLowerCase())])].filter(Boolean);
  if (!aliases.includes(`${en.toLowerCase()}s`)) aliases.push(`${en.toLowerCase()}s`);
  SYMBOLS[id] = {
    id,
    en: en.replace(/^\w/, (ch) => ch.toUpperCase()),
    zh: raw.zh || en,
    aliases,
    body: raw.body,
    kind: "icon",
    custom: true,
  };
  const saved = CATEGORIES.find((cat) => cat.id === "saved");
  if (saved && !BUILTIN_IDS.has(id) && !saved.ids.includes(id)) saved.ids.unshift(id);
  rebuildAliases();
  if (persist) persistCustom();
  return SYMBOLS[id];
}

export function loadCustomSymbols(list, { persist = false } = {}) {
  if (!Array.isArray(list)) return;
  for (const item of list) addCustomSymbol(item, { persist: false });
  if (persist) persistCustom();
}

export function readStoredSymbols() {
  if (typeof localStorage === "undefined") return [];
  try {
    const parsed = JSON.parse(localStorage.getItem(STORE) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

loadCustomSymbols(readStoredSymbols());

function tokens(text) {
  return String(text || "")
    .toLowerCase()
    .match(/\d+|[a-z]+|[\u4e00-\u9fff]/g) || [];
}

export function matchPrompt(text) {
  const found = [];
  const counts = {};
  const raw = String(text || "").toLowerCase();
  const words = tokens(text);
  for (let i = 0; i < words.length; i += 1) {
    const word = words[i];
    if (STOP.has(word)) continue;
    const hit = ALIAS.find((item) => item.alias === word);
    if (!hit) continue;
    const n = quantityAround(words, i);
    counts[hit.id] = (counts[hit.id] || 0) + n;
  }
  for (const { alias, id } of ALIAS) {
    if (!/[\u4e00-\u9fff]/.test(alias)) continue;
    const pattern = new RegExp(`(\\d+|[一二两三四五六七八九十])\\s*[只条个位辆朵棵座本杯艘台人]?\\s*${alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}|${alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*(\\d+)`, "g");
    let match;
    let total = 0;
    while ((match = pattern.exec(raw))) {
      total += parseCount(match[1] || match[2]) || 1;
    }
    if (!total) {
      const n = raw.split(alias).length - 1;
      if (n > 0 && !counts[id]) total = n;
    }
    if (total) counts[id] = Math.max(counts[id] || 0, total);
  }
  for (const [id, count] of Object.entries(counts)) found.push({ id, count: Math.min(12, count) });
  return found;
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function isWordChar(ch) {
  return /[a-zA-Z0-9]/.test(ch || "");
}

export function highlightPrompt(text) {
  const src = String(text || "");
  let i = 0;
  let html = "";
  while (i < src.length) {
    const lower = src.slice(i).toLowerCase();
    const hit = ALIAS.find((item) => {
      if (!lower.startsWith(item.alias)) return false;
      const before = src[i - 1];
      const after = src[i + item.alias.length];
      if (/[a-z]/.test(item.alias[0]) && isWordChar(before)) return false;
      if (/[a-z]/.test(item.alias.at(-1)) && isWordChar(after)) return false;
      return true;
    });
    if (hit) {
      const body = escapeHtml(src.slice(i, i + hit.alias.length));
      html += isRejected(hit.id) ? `<span class="unk">${body}</span>` : body;
      i += hit.alias.length;
      continue;
    }
    const token = src.slice(i).match(/^\d+|^[a-zA-Z]+|^[\u4e00-\u9fff]/);
    if (token) {
      const key = token[0].toLowerCase();
      const known = STOP.has(key) || UNIT.has(key) || NUM[key] || /^\d+$/.test(key);
      const body = escapeHtml(token[0]);
      html += known ? body : `<span class="unk">${body}</span>`;
      i += token[0].length;
      continue;
    }
    html += escapeHtml(src[i]);
    i += 1;
  }
  return html;
}

export function unknownWords(text) {
  const seen = new Set();
  const found = [];
  for (const word of tokens(text)) {
    if (STOP.has(word) || UNIT.has(word) || parseCount(word) || /^\d+$/.test(word)) continue;
    const aliasHit = ALIAS.find((item) => item.alias === word);
    if (aliasHit && !isRejected(aliasHit.id)) continue;
    if (seen.has(word)) continue;
    seen.add(word);
    found.push(word);
  }
  return found;
}

export function leftoverPrompt(text) {
  let next = ` ${String(text || "")} `;
  const matched = matchPrompt(text);
  for (const { id } of matched) {
    if (isRejected(id)) continue;
    for (const alias of SYMBOLS[id].aliases.sort((a, b) => b.length - a.length)) {
      next = next.replace(new RegExp(alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"), " ");
    }
  }
  return next.replace(/\s+/g, " ").trim();
}

export function symbolList() {
  return Object.values(SYMBOLS);
}
