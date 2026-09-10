import csv
from collections import defaultdict, Counter

BP = 'C:/Users/xl/OneDrive - Universitat Ramón Llull/bot_paper/'
PR = 'C:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/'

def gk(s):
    """Normalize gvkey to integer-string (strips zero-pad and .0)."""
    s = (s or '').strip()
    if s == '':
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return s

# ============================================================
# STEP 0: Load Compustat A+I (SIC, loc) — first-seen per gvkey
# ============================================================
sic = {}; loc = {}; fic = {}
n_cp_rows = 0
with open(BP+'compustat_panel.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        n_cp_rows += 1
        g = gk(row['gvkey'])
        s = (row['sic'] or '').strip()
        # sic may carry trailing .0
        if s.endswith('.0'): s = s[:-2]
        sic.setdefault(g, s)
        loc.setdefault(g, (row['loc'] or '').strip())
        fic.setdefault(g, (row['fic'] or '').strip())
print(f'[compustat_panel] rows={n_cp_rows} distinct_gvkeys={len(sic)}')

def is_fund(g):
    return sic.get(g) in ('6722', '6726')

# ============================================================
# STEP 1: robots universe from analysis_sample (gvkey, host, fund_flag, loc)
# ============================================================
as_gv2host = {}
as_fundflag = {}
as_loc = {}; as_fic = {}
as_univ = set()
with open(BP+'analysis_sample.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        g = gk(row['gvkey'])
        as_univ.add(g)
        as_gv2host[g] = (row['host'] or '').strip().lower()
        as_fundflag[g] = (row.get('fund_flag') or '').strip()
        as_loc[g] = (row.get('loc') or '').strip()
        as_fic[g] = (row.get('fic') or '').strip()
print(f'[analysis_sample] gvkeys={len(as_univ)} hosts={len(set(as_gv2host.values()))}')

# ============================================================
# CHECK A: does compustat SIC 6722/6726 reproduce analysis_sample fund_flag?
# ============================================================
ff1 = set(g for g in as_univ if as_fundflag.get(g) == '1')
sicfund = set(g for g in as_univ if is_fund(g))
print(f'[CHECK A] analysis_sample fund_flag==1 : {len(ff1)}')
print(f'[CHECK A] compustat SIC 6722/6726 (in univ): {len(sicfund)}')
print(f'[CHECK A]   in fund_flag but NOT sic: {len(ff1-sicfund)}; in sic but NOT fund_flag: {len(sicfund-ff1)}')
# breakdown
c6722 = sum(1 for g in as_univ if sic.get(g)=='6722')
c6726 = sum(1 for g in as_univ if sic.get(g)=='6726')
print(f'[CHECK A]   SIC 6722 in univ: {c6722}; SIC 6726 in univ: {c6726}; sum={c6722+c6726}')

# how many gvkeys have NO compustat match at all
nomatch = sum(1 for g in as_univ if g not in sic)
print(f'[CHECK A]   analysis_sample gvkeys with NO compustat_panel match: {nomatch}')

# ============================================================
# STEP 2: Build CLEAN sample from analysis_sample base
#   (1) drop funds (SIC 6722/6726)
#   (2) US/North-America (loc in USA/CAN) -- use analysis_sample loc
#   (3) host-dedup keep-min-gvkey
# ============================================================
op_gv = set(g for g in as_univ if not is_fund(g))
print(f'\n[CLEAN-AS base] after fund-drop: gvkeys={len(op_gv)} hosts={len(set(as_gv2host[g] for g in op_gv))}')
print(f'   funds dropped: {len(sicfund)} -> hosts among funds: {len(set(as_gv2host[g] for g in sicfund))}')

# US/NA using analysis_sample loc
op_na = set(g for g in op_gv if as_loc.get(g) in ('USA','CAN'))
print(f'[CLEAN-AS base] +US/NA(loc): gvkeys={len(op_na)} hosts={len(set(as_gv2host[g] for g in op_na))}')

# host-dedup keep min gvkey
host_best = {}
for g in sorted(op_na, key=lambda x: int(x)):
    h = as_gv2host[g]
    host_best.setdefault(h, g)
clean_gv = set(host_best.values())
clean_hosts = set(host_best.keys())
print(f'[CLEAN-AS base] +host-dedup: firms={len(clean_gv)} hosts={len(clean_hosts)}')

# ============================================================
# STEP 3: Load bot_family timing (full superset)
# ============================================================
bf = {}
with open(PR+'bot_family_specific_treatments.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        bf[gk(row['gvkey'])] = row
bf_all = set(bf.keys())
print(f'\n[bot_family] gvkeys={len(bf_all)}')
print(f'   clean_gv present in bf: {len(clean_gv & bf_all)} / {len(clean_gv)} (missing {len(clean_gv-bf_all)})')

# ============================================================
# FACT 2: train-only / search-only / both ; train-first/search-first/same
# ============================================================
def fact2(gvset, train, search):
    to=so=bo=tf=sf=sm=0
    for g in gvset:
        r = bf.get(g)
        if not r: continue
        t = r[train+'_canon_treated'] == '1'
        s = r[search+'_canon_treated'] == '1'
        if t and not s: to += 1
        elif s and not t: so += 1
        elif t and s:
            bo += 1
            tb = r[train+'_first_block'].strip(); sb = r[search+'_first_block'].strip()
            if tb and sb:
                if tb < sb: tf += 1
                elif sb < tb: sf += 1
                else: sm += 1
    return dict(train_only=to, search_only=so, both=bo, train_first=tf, search_first=sf, same_month=sm)

PAIRS = {'MAIN (gptbot vs oai_search)': ('gptbot','oai_search'),
         'BROAD (training_llm_core vs core_ai_search)': ('training_llm_core','core_ai_search')}

print('\n===== FACT 2 =====')
for lab,(tr,se) in PAIRS.items():
    print('--', lab)
    print('   OLD full bf universe  :', fact2(bf_all, tr, se))
    print('   OLD analysis_sample   :', fact2(as_univ, tr, se))
    print('   funds-only (as_univ)  :', fact2(sicfund, tr, se))
    print('   NEW clean (AS base)   :', fact2(clean_gv, tr, se))

# ============================================================
# STEP 4: Build CLEAN sample from the FULL bf base (the other derive agent's anchor)
#   universe = bf_all; need host + loc for these. analysis_sample only covers 9484.
#   Use compustat_all_host_mapping for host, compustat loc for US/NA.
# ============================================================
hostmap = {}
with open(BP+'compustat_all_host_mapping.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        hostmap[gk(row['gvkey'])] = (row['host_clean'] or '').strip().lower()
print(f'\n[host_mapping] gvkeys={len(hostmap)}')

# full-bf clean build
bf_op = set(g for g in bf_all if not is_fund(g))
bf_op_na = set(g for g in bf_op if loc.get(g) in ('USA','CAN'))
# host dedup using hostmap (fall back: gvkey itself if no host)
host_best_full = {}
no_host = 0
for g in sorted(bf_op_na, key=lambda x:int(x)):
    h = hostmap.get(g)
    if not h:
        no_host += 1
        continue
    host_best_full.setdefault(h, g)
clean_gv_full = set(host_best_full.values())
print(f'[CLEAN-bf base] fund-drop: {len(bf_op)} | +US/NA: {len(bf_op_na)} | gvkeys w/o host: {no_host} | after dedup firms={len(clean_gv_full)} hosts={len(host_best_full)}')

print('\n===== FACT 2 (full-bf clean base) =====')
for lab,(tr,se) in PAIRS.items():
    print('--', lab)
    print('   NEW clean (bf base, deduped):', fact2(clean_gv_full, tr, se))

# ============================================================
# FACT 1: named-AI-training-block + Googlebot allowed
# ============================================================
gv_bot = defaultdict(lambda: defaultdict(lambda: {'spec':0,'root':0,'gbb':None}))
host_bot = defaultdict(lambda: defaultdict(lambda: {'spec':0,'root':0,'gbb':None}))
host_gvk = defaultdict(set)
with open(BP+'robots_bot_panel.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        g = gk(row['gvkey']); h = (row['host'] or '').strip().lower(); b = row['bot']
        host_gvk[h].add(g)
        gbbv = row['googlebot_block']
        for store,key in ((gv_bot,g),(host_bot,h)):
            d = store[key][b]
            if row['specific_rule']=='1': d['spec']=1
            if row['root_block']=='1': d['root']=1
            if gbbv in ('0','1'): d['gbb'] = gbbv

trio = ('GPTBot','Google-Extended','ClaudeBot')
def gbb_of(store,key):
    # googlebot_block is host-level; pull from any bot row that carries it
    for b in store[key]:
        v = store[key][b]['gbb']
        if v is not None:
            return v
    return None

def fact1(store, keys, flag='spec'):
    blockers = [k for k in keys if any(store[k][b][flag] for b in trio)]
    n = len(blockers)
    op = sum(1 for k in blockers if gbb_of(store,k)=='0')
    bl = sum(1 for k in blockers if gbb_of(store,k)=='1')
    none = sum(1 for k in blockers if gbb_of(store,k) is None)
    return dict(named_blockers=n, gb_allowed=op, gb_blocked=bl, gb_unknown=none,
                pct=round(100*op/n,1) if n else None)

robots_hosts = set(host_gvk.keys())
print('\n===== FACT 1 (specific_rule named block) =====')
print('   OLD all firms gvkey :', fact1(gv_bot, as_univ))
print('   OLD all firms host  :', fact1(host_bot, robots_hosts))
print('   funds-only gvkey    :', fact1(gv_bot, sicfund))
print('   NEW clean gvkey     :', fact1(gv_bot, clean_gv))
print('   NEW clean host      :', fact1(host_bot, clean_hosts))

# robustness: use root_block (any block) instead of specific_rule
print('\n===== FACT 1 (root_block any-block variant) =====')
print('   OLD all firms gvkey :', fact1(gv_bot, as_univ, 'root'))
print('   NEW clean gvkey     :', fact1(gv_bot, clean_gv, 'root'))

# identify the 3 Googlebot-blocked named blockers in clean sample
print('\n--- clean-sample named blockers that ALSO block Googlebot ---')
for g in sorted(clean_gv, key=lambda x:int(x)):
    if any(gv_bot[g][b]['spec'] for b in trio) and gbb_of(gv_bot,g)=='1':
        # get firm name
        print('   gvkey', g, 'host', as_gv2host.get(g))
