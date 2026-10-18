const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../src/scada_rag/web/app.js'),'utf8');

const status=()=>({ready:true,busy:false,answering:false,logs:[],error:null,documents:16,chunks:173,
  config:{llm_model:'qwen2.5:3b'},ollama:{running:true,model_ready:true},sources:[]});
const response=value=>({ok:true,status:200,json:async()=>value});

function element(){return {textContent:'',value:'',disabled:false,hidden:false,dataset:{},children:[],options:[{}],listeners:{},
  classList:{toggle(){}},append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},
  addEventListener(name,fn){this.listeners[name]=fn;},add(item){this.options.push(item);},remove(i){this.options.splice(i,1);},focus(){}};}

async function app(fetch){
  const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
  const context=vm.createContext({document:{getElementById:get,querySelectorAll:()=>[],createElement:element},
    fetch,AbortController,URL,TypeError,setTimeout,clearTimeout,setInterval:()=>0,clearInterval:()=>{},
    Option:function(title,id){this.text=title;this.value=id;}});
  vm.runInContext(source,context);
  await new Promise(resolve=>setImmediate(resolve));
  return {context,get};
}

test('missing Ollama model is distinguished from a Python server outage',async()=>{
  const s=status();s.ollama={running:true,model_ready:false,error:'ollama pull qwen2.5:3b'};
  const {get}=await app(async()=>response(s));
  assert.match(get('status').textContent,/Не установлена модель/);
  assert.equal(get('send').disabled,true);
  assert.match(get('logs').textContent,/ollama pull qwen2.5:3b/);
  assert.equal(get('connection-error').hidden,false);
  assert.match(get('connection-error').textContent,/ollama pull qwen2.5:3b/);
  assert.equal(get('setup').hidden,true);
});

test('losing the Python server disables sending another question',async()=>{
  let offline=false;
  const {get,context}=await app(async()=>{if(offline)throw new TypeError('Failed to fetch');return response(status());});
  assert.equal(get('send').disabled,false);
  offline=true;await vm.runInContext('refresh()',context);
  assert.equal(get('status').textContent,'Нет связи с сервером Python');
  assert.equal(get('send').disabled,true);
});

test('a stalled request is aborted and reports a timeout',async()=>{
  const {context}=await app(async(url,options)=>{
    if(url!=='/api/ask')return response(status());
    return new Promise((_,reject)=>options.signal.addEventListener('abort',()=>{
      const error=new Error('aborted');error.name='AbortError';reject(error);
    },{once:true}));
  });
  await assert.rejects(vm.runInContext("api('/api/ask',{}, {timeoutMs:20})",context),/Превышено время ожидания/);
});

test('a malformed source URL does not mark the Python server as unavailable',async()=>{
  const s=status();s.sources=[{id:'x',title:'Документ',url:':invalid'}];
  const {get}=await app(async()=>response(s));
  assert.equal(get('status').textContent,'Система готова');
  assert.equal(get('source-list').children.length,1);
});

test('an Ollama error is shown and the question button returns to its normal state',async()=>{
  const {get,context}=await app(async url=>url==='/api/ask'?{ok:false,status:500,json:async()=>({error:'Ollama не отвечает'})}:response(status()));
  get('question').value='Что такое SCADA?';get('top-k').value='20';get('threshold').value='0.72';
  await get('question-form').listeners.submit({preventDefault(){}});
  assert.equal(vm.runInContext('asked',context),false);
  assert.equal(get('send').disabled,false);
  assert.equal(get('send').textContent,'Задать вопрос ↗');
  assert.equal(get('conversation').children.at(-1).children[0].textContent,'Ollama не отвечает');
});
