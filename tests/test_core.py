import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scada_rag.preprocessing.text import clean, windows, chunk_document
from scada_rag.grabber.collector import parse_html, resolve_anre
from scada_rag.generation.ollama import REFUSAL, validate_answer, generate, messages
from scada_rag.reranking.ranker import filter_hits
from scada_rag.evaluation.metrics import retrieval_metrics, validate_evidence
from scada_rag.common import write_json, read_json


DOC = {"id":"source","source":"test","title":"Документ","url":"https://example.org/a","language":"ru", "updated_at":"2026-10-18",
       "blocks":[{"section":"Первый раздел","text":"Важные сведения о системе и её архитектуре. "*25,"page":1},
                 {"section":"Второй раздел","text":"Другие сведения о работе системы и доступе. "*25,"page":2}]}
HIT = {"source_id":"source","text":"Полезный текст про защищённую систему управления и сбор данных.","similarity":0.85}


class CoreTests(unittest.TestCase):
    def test_chunk_size_overlap_and_coverage(self):
        text = ''.join(chr(0x400+i) for i in range(500))
        parts = list(windows(text,100,20))
        self.assertEqual(parts[0][-20:],parts[1][:20])
        self.assertEqual(parts[0] + ''.join(p[20:] for p in parts[1:]),text)
        self.assertTrue(all(len(p)<=100 for p in parts))

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError): list(windows('abc',10,10))

    def test_sections_preserve_page_and_stable_ids(self):
        chunks=chunk_document(DOC,'sections',250,30)
        self.assertEqual(chunks,chunk_document(DOC,'sections',250,30))
        self.assertEqual({c['page'] for c in chunks},{1,2})
        self.assertTrue(all(len(c['text'])<=250 for c in chunks))

    def test_normalization(self):
        self.assertEqual(clean('систе-\nма\u00a0  управления\u00ad'),'система управления')

    def test_html_drops_navigation(self):
        raw='<html><title>x</title><nav>НЕ НУЖНО</nav><main><nav>ЕЩЕ МЕНЮ</nav><h1>Тема</h1><h2>Секция</h2><p>'+('Русский текст документа. '*20)+'</p></main></html>'
        title,blocks=parse_html(raw.encode())
        self.assertEqual(title,'Тема')
        self.assertNotIn('МЕНЮ',str(blocks))
        self.assertEqual(blocks[0]['section'],'Секция')

    def test_anre_unique_resolution(self):
        raw=b'<div class="block-activity-report"><div class="title">HCA nr. 586 din 16.10.2024</div><a class="download-link" href="/a.pdf">PDF</a></div>'
        with patch('scada_rag.grabber.collector.fetch',return_value=(raw,{},'')):
            self.assertEqual(resolve_anre({'registry':'https://anre.md/registry','match':'586 din 16.10.2024'}),'https://anre.md/a.pdf')

    def test_no_context_never_calls_llm(self):
        with patch('scada_rag.generation.ollama.request_json') as request:
            result=generate('Вопрос',[],{})
            request.assert_not_called()
            self.assertEqual(result['answer'],REFUSAL)

    def test_invalid_and_missing_citations_refuse(self):
        for ids in [[],[2],[True],['1']]:
            self.assertFalse(validate_answer({'answerable':True,'answer':'Ответ','citations':ids},[HIT])['answerable'])

    def test_valid_citations_refer_to_context(self):
        answer=validate_answer({'answerable':True,'answer':'Ответ','citations':[1,1]},[HIT])
        self.assertTrue(answer['answerable']);self.assertEqual(len(answer['sources']),1)

    def test_filter_threshold_and_dedup(self):
        self.assertEqual(len(filter_hits([HIT,HIT,{**HIT,'similarity':0.1}],0.7)),1)
        self.assertEqual(filter_hits([{**HIT,'reranker_score':0.01}],0.7,reranker_threshold=0.5),[])

    def test_retrieval_requires_evidence_not_just_document(self):
        q={'evidence':[{'source_id':'source','contains':['сбор данных']}]}
        score=retrieval_metrics([HIT,{**HIT,'text':'другая тема'}],q,2)
        self.assertEqual(score['precision'],0.5);self.assertEqual(score['recall'],1)
        self.assertEqual(retrieval_metrics([{**HIT,'text':'другая тема'}],q,1)['recall'],0)

    def test_multi_document_recall(self):
        q={'evidence':[{'source_id':'source','contains':['сбор данных']},{'source_id':'another','contains':['доступ']}]}
        self.assertEqual(retrieval_metrics([HIT],q,3)['recall'],0.5)

    def test_evidence_drift_detected(self):
        self.assertEqual(len(validate_evidence([{'id':'q','evidence':[{'source_id':'missing','contains':['a']}]}],[DOC])),1)

    def test_refusal_does_not_expose_sources(self):
        self.assertEqual(validate_answer({'answerable':False,'answer':'любой','citations':[1]},[HIT])['sources'],[])

    def test_atomic_json_unicode(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'папка'/'данные.json';write_json(path,{'текст':'SCADA'});self.assertEqual(read_json(path),{'текст':'SCADA'})

    def test_faiss_metadata_filter_and_unicode_path(self):
        import faiss
        from scada_rag.retrieval.store import VectorStore
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'кириллица';path.mkdir()
            idx=faiss.IndexFlatIP(2);idx.add(np.array([[1,0],[0.8,0.6],[0,1]],dtype='float32'))
            (path/'vectors.faiss').write_bytes(faiss.serialize_index(idx).tobytes())
            write_json(path/'metadata.json',{'complete':True});write_json(path/'chunks.json',[{**HIT,'source_id':str(i)} for i in range(3)])
            store=VectorStore(path)
            self.assertEqual(store.search(np.array([[1,0]],dtype='float32'),1,'2')[0]['source_id'],'2')

    def test_prompt_treats_documents_as_untrusted(self):
        self.assertIn('недоверенные',messages('q',[{**HIT,'title':'a','section':'b'}])[0]['content'])

    def test_grabber_twice_no_duplicates(self):
        from scada_rag.grabber.collector import collect
        raw=('<main><h1>Тема</h1><p>'+'Данные документа и описание SCADA. '*15+'</p></main>').encode()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'configs/sources.json',{'sources':[{'id':'x','kind':'html','url':'https://example.org/doc'}]})
            with patch('scada_rag.grabber.collector.ROOT',root),patch('scada_rag.grabber.collector.DATA',root/'data'),patch('scada_rag.grabber.collector.fetch',return_value=(raw,{},'https://example.org/doc')),patch('scada_rag.grabber.collector.time.sleep'):
                self.assertEqual(collect(lambda _:None)[0]['status'],'new')
                self.assertEqual(collect(lambda _:None)[0]['status'],'unchanged')
                self.assertEqual(len(list((root/'data/processed').glob('*.json'))),1)

    def test_grabber_error_keeps_last_good_document(self):
        from scada_rag.grabber.collector import collect
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'configs/sources.json',{'sources':[{'id':'x','kind':'html','url':'https://example.org/doc'}]})
            path=root/'data/processed/x.json';write_json(path,DOC)
            with patch('scada_rag.grabber.collector.ROOT',root),patch('scada_rag.grabber.collector.DATA',root/'data'),patch('scada_rag.grabber.collector.fetch',side_effect=OSError('network')),patch('scada_rag.grabber.collector.time.sleep'):
                self.assertEqual(collect(lambda _:None)[0]['status'],'error')
                self.assertEqual(read_json(path),DOC)

    def test_http_local_ui_and_api(self):
        from http.server import ThreadingHTTPServer
        from scada_rag.server import make_handler
        class App:
            def __init__(self): self.state={'ready':False}
            def status(self): return {'ready':False,'documents':0}
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(App()))
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            with urllib.request.urlopen(base+'/') as r:self.assertIn('SCADA',r.read().decode())
            with urllib.request.urlopen(base+'/api/status') as r:self.assertEqual(json.load(r)['documents'],0)
            request=urllib.request.Request(base+'/api/ask',data=b'{"question":"q"}',headers={'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(request)
            self.assertEqual(error.exception.code,409)
        finally:server.shutdown();server.server_close();worker.join()


if __name__=='__main__': unittest.main()
