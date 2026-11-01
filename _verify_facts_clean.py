import csv
from collections import defaultdict

BP = 'C:/Users/xl/OneDrive - Universitat Ramón Llull/bot_paper/'
PR = 'C:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/'

def gk(s):
    s = (s or '').strip()
    if s == '':
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return s

# ---------- load compustat SIC (A+I) ----------
sic = {}; loc = {}; fic = {}
with open(BP+'compustat_panel.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        g = gk(row['gvkey'])
        sic.setdefault(g, (row['sic'] or '').replace('.0',''))
        loc.setdefault(g, row['loc']); fic.setdefault(g, row['fic'])
def is_fund(g):
    return sic.get(g) in ('6722', '6726')

# ---------- robots universe (gvkey,host) from analysis_sample ----------
g2host = {}
univ = set()
with open(BP+'analysis_sample.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        g = gk(row['gvkey'])
        univ.add(g); g2host[g] = row['host']

print('=== UNIVERSE (robots) ===')
print('gvkeys:', len(univ), '| hosts:', len(set(g2host.values())))
fund_gv = set(g for g in univ if is_fund(g))
op_gv = set(g for g in univ if not is_fund(g))
print('fund gvkeys (SIC 6722/6726):', len(fund_gv), '-> hosts:', len(set(g2host[g] for g in fund_gv)))
print('operating gvkeys:', len(op_gv), '-> hosts:', len(set(g2host[g] for g in op_gv)))

# ---------- CLEAN SAMPLE: drop funds, US/NA, host-dedup ----------
op_na = set(g for g in op_gv if loc.get(g) in ('USA','CAN'))
# host dedup -> keep min gvkey per host
host_best = {}
for g in sorted(op_na, key=lambda x: int(x)):
    h = g2host[g]
    if h not in host_best:
        host_best[h] = g
clean_gv = set(host_best.values())
clean_hosts = set(host_best.keys())
print('\n=== CLEAN SAMPLE ===')
print('operating non-fund US/NA gvkeys:', len(op_na), '| hosts:', len(set(g2host[g] for g in op_na)))
print('after host-dedup -> firms:', len(clean_gv), '| hosts:', len(clean_hosts))

# ---------- load bot_family timing ----------
bf = {}
with open(PR+'bot_family_specific_treatments.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        bf[gk(row['gvkey'])] = row
bf_all = set(bf.keys())

# ---------- FACT 2 ----------
def fact2(gvset, train, search):
    to=so=bo=tf=sf=sm=0
    for g in gvset:
        r = bf.get(g)
        if not r: continue
        t = r[train+'_canon_treated']=='1'
        s = r[search+'_canon_treated']=='1'
        if t and not s: to+=1
        elif s and not t: so+=1
        elif t and s:
            bo+=1
            tb=r[train+'_first_block'].strip(); sb=r[search+'_first_block'].strip()
            if tb and sb:
                if tb<sb: tf+=1
                elif sb<tb: sf+=1
                else: sm+=1
    return dict(train_only=to, search_only=so, both=bo, train_first=tf, search_first=sf, same_month=sm)

print('\n=== FACT 2: TRAIN-BEFORE-RETRIEVAL ===')
for lab, (tr, se) in {'MAIN (gptbot vs oai_search)':('gptbot','oai_search'),
                      'BROAD (training_llm_core vs core_ai_search)':('training_llm_core','core_ai_search')}.items():
    print('--', lab)
    print('  OLD full bf universe (all firms incl funds):', fact2(bf_all, tr, se))
    print('  OLD robots universe (all firms incl funds) :', fact2(univ, tr, se))
    print('  NEW clean op-co (deduped)                  :', fact2(clean_gv, tr, se))
    # fund contribution to OLD both/train_only
    print('  (funds-only subset)                        :', fact2(fund_gv, tr, se))

# ---------- FACT 1 (snapshot named training block + Googlebot allowed) ----------
gv_bot = defaultdict(lambda: defaultdict(lambda: {'spec':0,'root':0,'gbb':0}))
host_bot = defaultdict(lambda: defaultdict(lambda: {'spec':0,'root':0,'gbb':0}))
with open(BP+'robots_bot_panel.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        g = gk(row['gvkey']); h = row['host']; b = row['bot']
        for store, key in ((gv_bot, g), (host_bot, h)):
            d = store[key][b]
            if row['specific_rule']=='1': d['spec']=1
            if row['root_block']=='1': d['root']=1
            if row['googlebot_block']=='1': d['gbb']=1

trio = ('GPTBot','Google-Extended','ClaudeBot')
def fact1(store, keys):
    blockers = [k for k in keys if any(store[k][b]['spec'] for b in trio)]
    n = len(blockers)
    gb_open = sum(1 for k in blockers if store[k]['Googlebot']['gbb']==0)
    gb_blk  = sum(1 for k in blockers if store[k]['Googlebot']['gbb']==1)
    return dict(named_blockers=n, googlebot_allowed=gb_open, googlebot_blocked=gb_blk,
                pct=round(100*gb_open/n,1) if n else None)

print('\n=== FACT 1: AI-vs-SEARCH SEPARATION (named GPTBot/Google-Extended/ClaudeBot block; Googlebot allowed) ===')
print('  OLD all firms, gvkey-level:', fact1(gv_bot, univ))
print('  OLD all firms, host-level :', fact1(host_bot, set(g2host.values())))
print('  funds-only,    gvkey-level:', fact1(gv_bot, fund_gv))
print('  NEW clean,     host-level :', fact1(host_bot, clean_hosts))
print('  NEW clean,     gvkey-level:', fact1(gv_bot, clean_gv))
