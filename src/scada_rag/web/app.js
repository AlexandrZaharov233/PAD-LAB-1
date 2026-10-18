const $=id=>document.getElementById(id);let state={},asked=false,pollBusy=false;
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function connectionError(message){$('connection-error').hidden=!message;$('connection-error').textContent=message||'';}
async function api(url,data,{timeoutMs=10000}={}){
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),timeoutMs);
  try{
    const r=await fetch(url,{...(data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}),signal:controller.signal});
    let value;
    try{value=await r.json();}catch(e){if(e.name==='AbortError')throw e;throw Error('Сервер вернул некорректный ответ. Проверьте окно start.bat.');}
    if(!r.ok)throw Error(value?.error||'Ошибка сервера: HTTP '+r.status);
    return value;
  }catch(e){
    if(e.name==='AbortError')throw Error(url==='/api/ask'?'Превышено время ожидания ответа. Проверьте окно start.bat: загрузка модели могла ещё продолжаться.':'Сервер не ответил за 10 секунд. Проверьте окно start.bat.');
    if(e instanceof TypeError)throw Error('Нет соединения с сервером Python. Проверьте, что окно start.bat открыто.');
    throw e;
  }finally{clearTimeout(timer);}
}
document.querySelectorAll('.nav').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.nav').forEach(n=>n.classList.toggle('active',n===b));document.querySelectorAll('.tab').forEach(t=>t.hidden=t.id!==b.dataset.tab);$('page-title').textContent=b.textContent;if(b.dataset.tab==='experiments')loadResults();}));
$('top-k').oninput=()=>{$('k-value').textContent=$('top-k').value;};$('threshold').oninput=()=>{$('threshold-value').textContent=Number($('threshold').value).toFixed(2);};
document.querySelectorAll('.examples button').forEach(b=>b.onclick=()=>{$('question').value=b.textContent;$('question').focus();});
function sourceCard(x,citation){const box=el('div',undefined,'source-card');const a=el('a',(citation?'['+citation+'] ':'')+x.title);try{const url=new URL(x.url);if(['http:','https:'].includes(url.protocol)){if(x.page)url.hash='page='+x.page;a.href=url.href;a.target='_blank';a.rel='noopener noreferrer';}}catch{}box.append(a);box.append(el('div',[x.source,x.section,x.page?'стр. PDF '+x.page:'',x.similarity!==undefined?'сходство '+x.similarity.toFixed(3):''].filter(Boolean).join(' · '),'source-meta'));if(x.note)box.append(el('p',x.note));if(x.text){const detail=el('details');detail.append(el('summary','Показать фрагмент'));detail.append(el('p',x.text));box.append(detail);}return box;}
async function refresh(){
  if(pollBusy)return;pollBusy=true;
  try{
    try{state=await api('/api/status');}
    catch(e){$('status').textContent='Нет связи с сервером Python';$('status').title=e.message;$('status').classList.toggle('ready',false);$('send').disabled=true;$('model-line').textContent=e.message;connectionError(e.message);return;}
    try{
      const ready=state.ready&&state.ollama.model_ready,answering=asked||state.answering;
      $('status').textContent=state.busy?'Подготовка базы...':!state.ollama.running?'Ollama недоступна':!state.ollama.model_ready?'Не установлена модель '+state.config.llm_model:state.error?'Ошибка подготовки базы':!state.ready?'База не подготовлена':answering?'Обработка вопроса...':'Система готова';
      $('status').title=state.ollama.error||state.error||'';$('status').classList.toggle('ready',ready&&!state.busy);
      $('setup').hidden=state.ready&&!state.busy&&!state.error;
      connectionError(state.ollama.error||'');
      $('logs').textContent=state.logs.join('\n');
      if(state.error)$('logs').textContent+='\n'+state.error;
      if(state.ollama.error)$('logs').textContent+='\n'+state.ollama.error;
      $('prepare').disabled=state.busy||answering;$('refresh').disabled=state.busy||answering;
      $('prepare').textContent=state.busy?'Подготовка...':'Подготовить документы';
      $('doc-count').textContent=state.documents;$('chunk-count').textContent=state.chunks;
      $('model-line').textContent='Генерация: '+state.config.llm_model+' · '+(state.ollama.running?(state.ollama.model_ready?'Ollama запущена':'скачайте модель через ollama pull'):'запустите Ollama');
      $('send').disabled=!ready||state.busy||answering;
      const signature=JSON.stringify(state.sources);
      if($('source-list').dataset.signature!==signature){const select=$('source-filter'),selected=select.value;while(select.options.length>1)select.remove(1);$('source-list').replaceChildren();state.sources.forEach(s=>{select.add(new Option(s.title,s.id));$('source-list').append(sourceCard(s));});select.value=selected;$('source-list').dataset.signature=signature;}
    }catch(e){$('status').textContent='Ошибка отображения состояния';$('status').title=e.message;$('send').disabled=true;$('logs').textContent=e.message;connectionError(e.message);}
  }finally{pollBusy=false;}
}
async function prepare(){try{await api('/api/prepare',{});await refresh();}catch(e){alert(e.message);}}$('prepare').onclick=prepare;$('refresh').onclick=prepare;
$('question-form').addEventListener('submit',async event=>{
  event.preventDefault();if(asked||$('send').disabled)return;
  const question=$('question').value.trim();if(!question)return;
  asked=true;$('welcome').hidden=true;$('send').disabled=true;$('send').textContent='Ищем ответ...';
  const conversation=$('conversation');conversation.append(el('div',question,'question-card'));
  const answer=el('div',undefined,'answer-card'),title=el('div','ПОИСК И ПОДГОТОВКА ОТВЕТА','answer-title');
  answer.append(title);answer.append(el('p','Идёт поиск и генерация ответа. Оставьте окно start.bat открытым.'));conversation.append(answer);
  const started=Date.now(),timer=setInterval(()=>{title.textContent='ПОДГОТОВКА ОТВЕТА · '+Math.floor((Date.now()-started)/1000)+' с';},1000);
  try{
    const result=await api('/api/ask',{question,settings:{top_k:Number($('top-k').value),similarity_threshold:Number($('threshold').value),use_reranker:$('reranker').checked,source_id:$('source-filter').value,prompt_variant:$('prompt').value}},{timeoutMs:360000});
    answer.replaceChildren();answer.append(el('div',result.answerable?'ОТВЕТ ПО ДОКУМЕНТАМ':'ИНФОРМАЦИИ НЕДОСТАТОЧНО','answer-title'));answer.append(el('div',result.answer,'answer-text'));
    result.sources.forEach(s=>answer.append(sourceCard(s,s.citation)));
    const d=el('details');d.append(el('summary','Найдено '+result.retrieved.length+' фрагментов · в контексте '+result.context.length));result.context.forEach(s=>d.append(sourceCard(s)));answer.append(d);
    answer.append(el('div','Поиск: '+result.retrieval_seconds.toFixed(1)+' с · Всего: '+result.total_seconds.toFixed(1)+' с','time'));$('question').value='';
  }catch(e){answer.replaceChildren(el('div',e.message,'error'));}
  finally{clearInterval(timer);asked=false;$('send').textContent='Задать вопрос ↗';await refresh();}
});
$('question').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();if(!$('send').disabled)$('question-form').requestSubmit();}});
async function loadResults(){
  try{
    const groups=await api('/api/results');$('results').replaceChildren();
    if(!groups.length){$('results').append(el('p','Эксперименты ещё не запускались. Результаты появятся после выполнения experiments.bat.'));return;}
    const names={baseline:'Базовая',fixed_600:'Фрагменты 600',fixed_1000:'Фрагменты 1000',fixed_1400:'Фрагменты 1400',overlap_0:'Перекрытие 0',overlap_150:'Перекрытие 150',overlap_300:'Перекрытие 300',paragraphs:'Абзацы',k3:'Поиск: 3',k5:'Поиск: 5',k10:'Поиск: 10',without_reranker:'Без переранжирования',reranker_threshold_0:'Порог переранжирования 0',threshold_068:'Порог сходства 0,68',threshold_080:'Порог сходства 0,80',e5_embedding_only:'Только E5',alternative_embedding_only:'Только MiniLM',prompt_concise:'Краткая инструкция',temperature_04:'Температура 0,4'};
    groups.forEach(g=>{
      const box=el('div',undefined,'result-group');box.append(el('div',g.path,'source-meta'));
      const columns=[['name','Конфигурация'],['context_recall','Recall контекста'],['context_mrr','MRR контекста'],['context_precision','Precision контекста'],['seconds_mean','Поиск, с']];
      if(g.rows.some(r=>r.refusal_correct!==undefined))columns.push(['refusal_correct','Ответить / отказать'],['keyword_coverage','Ключевые фразы'],['false_answer','Ложные ответы']);
      const table=el('table'),head=el('tr');columns.forEach(c=>head.append(el('th',c[1])));table.append(head);
      g.rows.forEach(r=>{const row=el('tr');columns.forEach(([key])=>{const v=key==='name'?(names[r.name]||r.name):r[key];row.append(el('td',typeof v==='number'?v.toFixed(3):(v??'—')));});table.append(row);});
      box.append(table);$('results').append(box);
    });
  }catch(e){$('results').replaceChildren(el('div',e.message,'error'));}
}
refresh();setInterval(refresh,3500);
