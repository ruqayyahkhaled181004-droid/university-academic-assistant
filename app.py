import os, uuid, json
import streamlit as st
from rag import AcademicAssistant, conversation_history

st.set_page_config(page_title='Campus Guide', page_icon='🎓', layout='wide')
st.title('🎓 Campus Guide')
st.caption('University Academic Assistant • Answers grounded in your uploaded documents')
st.info('Demo documents contain fictional policies. Use official university documents for real guidance.')
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
with st.sidebar:
    st.header('Knowledge sources')
    st.caption('Upload academic handbooks, course catalogs, or registration policies. Text is sent to Gemini for processing.')
    uploads = st.file_uploader('PDF or TXT documents', type=['pdf','txt'], accept_multiple_files=True)
    version = st.selectbox('Prompt version', ['v2','v1'])
    try:
        key = st.secrets.get('GEMINI_API_KEY','') or st.secrets.get('GOOGLE_API_KEY','')
    except Exception:
        key = ''
    key = key or os.getenv('GEMINI_API_KEY','') or os.getenv('GOOGLE_API_KEY','')
    if not key:
        key = st.text_input('Gemini API key', type='password')
    demo_mode = st.checkbox('Use included fictional demo documents')
    if st.button('Build knowledge index', type='primary'):
        from pathlib import Path
        selected = [(p.name,p.read_bytes()) for p in sorted(Path('demo_data').glob('*.txt'))] if demo_mode else [(f.name,f.getvalue()) for f in (uploads or [])]
        if not key or not selected:
            st.warning('Provide an API key and at least one document.')
        else:
            try:
                with st.spinner('Extracting, chunking and embedding documents…'):
                    bot = AcademicAssistant(key, f'logs/{st.session_state.session_id}.jsonl')
                    stats = bot.build(selected)
                    st.session_state.bot = bot
                    st.session_state.messages = []
                    st.session_state.stats = stats
                st.success('Index ready')
            except Exception as e:
                st.error(f'Index failed ({type(e).__name__}). Check readable text, file format and API quota.')
    if 'stats' in st.session_state:
        st.json(st.session_state.stats)
        st.caption('The sources listed above are the active index. Rebuild after changing uploads.')
    if st.button('Clear conversation'):
        st.session_state.messages = []
        st.rerun()

def show_result(r):
    st.markdown(r['answer'])
    st.caption(f"{r['category']} · {r['latency_s']} s · paid-tier estimate ${r['estimated_usd']:.6f} · {r['status']}")
    with st.expander('Sources and retrieved passages'):
        for i,c in enumerate(r['evidence'],1):
            st.markdown(f"**[S{i}] {c['source']} — page {c['page']}**")
            st.caption(f"{c['chunk_id']} • similarity {c['score']:.3f}")
            st.text(c['text'])
    with st.expander('Usage metrics'):
        st.json({k:r[k] for k in ['version','input_tokens','output_tokens','embedding_tokens','embedding_tokens_estimated','generation_usage_missing','citation_valid']})

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        if 'result' in msg:
            show_result(msg['result'])
        else:
            st.markdown(msg['content'])
if question := st.chat_input('Ask about courses, registration, attendance, exams or graduation'):
    if 'bot' not in st.session_state:
        st.warning('Build your knowledge index in the sidebar first.')
    else:
        history = conversation_history(st.session_state.messages)
        with st.chat_message('user'):
            st.markdown(question)
        with st.chat_message('assistant'):
            with st.spinner('Finding supporting evidence…'):
                result = st.session_state.bot.ask(question,history,version)
            show_result(result)
        st.session_state.messages.extend([{'role':'user','content':question},
            {'role':'assistant','content':result['answer'],'result':result}])

if 'bot' in st.session_state:
    with st.expander('Session monitoring dashboard'):
        p = st.session_state.bot.log_path
        if p.exists():
            events = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
            requests = [e for e in events if e.get('event')=='question']
            columns = st.columns(3)
            columns[0].metric('Questions',len(requests))
            columns[1].metric('Request errors',sum(e.get('status')=='error' for e in requests))
            columns[2].metric('Paid-tier equivalent USD',f"{sum(e.get('estimated_usd',0) for e in events):.6f}")
            st.caption('Includes indexing. Free-tier billed cost may be zero. Latency includes quota pacing and retries. Citation labels still require a support check.')
            st.dataframe([{k:e.get(k) for k in ['time','event','status','version','latency_s','input_tokens','output_tokens','embedding_tokens']} for e in events])
            st.download_button('Download session metrics',p.read_bytes(),'events.jsonl')

