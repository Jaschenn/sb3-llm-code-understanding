#!/usr/bin/env python
"""Issue #7 汇总脚本：由 docs/evidence/issues/issue-*.md 生成 ai-log.md 与 evidence-matrix.md。

只读 issue 文件，不联网、不运行上游代码。
执行（仓库根目录）：python3 -I scripts/issue_07_build_summary.py .
预期输出：19 ai rows 58 matrix rows（随 issue 文件增减变化）。
结论见 docs/evidence/issues/issue-07.md。
"""
import re, sys, pathlib
root = pathlib.Path(sys.argv[1]); ev = root/'docs/evidence'; iss = ev/'issues'
ISSUES = ['01','02','03','04','05','06','07','08']
HANDLE = {'成员03':'@langzim','成员01':'@Jaschenn'}
def fix(s):
    s = s.replace('下方', 'issue-01.md 中的')
    s = s.replace('../../../','../../')
    for k,v in HANDLE.items(): s = s.replace('@'+k, v).replace(k, v)
    return s.strip()
def rows(text, prefix):
    out=[]
    for l in text.splitlines():
        if re.match(r'\|\s*'+prefix, l):
            c=[x.strip() for x in l.strip().strip('|').split('|')]
            out.append(c)
    return out
def esc(s): return s.replace('\n',' ').replace('|','\\|') if False else s
def norm_status(s):
    if '需限定' in s or '但需限定' in s or '限定' in s.split('：')[0]: base='需限定'
    elif s.startswith('已否定'): base='已否定'
    elif s.startswith('已支持'): base='已支持'
    elif s.startswith('需限定'): base='需限定'
    else: base='待验证'
    return base
def split_ev(cell):
    src,tst,run=[],[],[]
    for seg in re.split(r'；|(?<=\))、(?=\[)', cell):
        seg=seg.strip()
        if not seg: continue
        if re.search(r'tests/|test_', seg) and not re.search(r'artifacts/|scripts/', seg): tst.append(seg)
        elif re.search(r'artifacts/|scripts/|运行|summary\.md|output', seg): run.append(seg)
        elif re.search(r'test', seg, re.I): tst.append(seg)
        else: src.append(seg)
    j=lambda a:'；'.join(a)
    return j(src),j(tst),j(run)
matrix=[]; ailog=[]
for n in ISSUES:
    t=(iss/f'issue-{n}.md').read_text()
    own=re.search(r'负责人[：:]\s*(@[\w-]+|@成员\d+)',t); rev=re.search(r'复核人[：:]\s*(@[\w-]+|@成员\d+)',t)
    who=fix(own.group(1)) if own else ''; rv=fix(rev.group(1)) if rev else ''
    pair = f'负责人：{who}；复核人：{rv}' if rv else f'负责人：{who}；复核人：待补'
    aiids=[r for r in rows(t,'AI-')]
    cs=rows(t,r'C-\d\d-\d+')
    for c in cs:
        cid=c[0]
        if n=='01':
            m=re.match(r'(已支持|已否定|需限定|待验证)[：:]?(.*)',c[2],re.S); status=m.group(1); evc=m.group(2)
            concl=c[1]; src,tst,run=split_ev(evc)
            if cid=='C-01-01': src='固定 commit 下运行 issue-01.md 中的统计命令（实际输出 64 / 15693）'
        elif n in('03','04'):
            concl=c[1]; status=norm_status(c[2]); src=c[3]; tst=''; run=c[4] if len(c)>4 else ''
            q=c[2].split('：',1)[1] if '：' in c[2] else ''
            if q: concl+=f'（{q}）'
        else:
            concl=c[1]; status=norm_status(c[2]); q=c[2]
            if q not in ('已支持','已否定','需限定','待验证'):
                concl+=f'（状态说明：{q}）'
            src,tst,run=split_ev(c[3])
        # AI refs: AI rows (other than 01) mentioning this C id
        refs=[]
        for a in aiids:
            if cid in ' '.join(a): refs.append(a[0])
        if n=='05':  # preserve cross-refs recorded by original T05 author
            pass
        matrix.append((cid,fix(concl),fix(src),fix(tst) or '见该 Issue 文件',fix(run),'、'.join(refs),status,pair,n))
    # AI rows
    if n=='01':
        for i,a in enumerate(rows(t,r'"[^|]*'),1):
            pass
        lines=[l for l in t.split('## AI 使用记录')[1].split('**纠错案例**')[0].splitlines() if l.startswith('|') and '"' in l]
        for i,l in enumerate(lines,1):
            c=[x.strip() for x in l.strip().strip('|').split('|')]
            res=c[3].replace('✅','').replace('⚠️','').strip()
            fixd=''; r=res
            if res.startswith('部分错误') and '：' in res: r,fixd=res.split('：',1)
            elif '：' in res: r=res; fixd=''
            ailog.append((f'AI-01-{i:02d}',c[0].strip('"'),c[1],c[2],r.strip(),fixd.strip() or '无修正',pair))
    else:
        for a in aiids:
            if len(a)>=6: ailog.append((a[0],a[1],a[2],a[3],a[4],a[5],pair))
# fix 01 matrix AI refs (original ids by content)
aimap={'C-01-02':'AI-01-03','C-01-03':'AI-01-04','C-01-01':'AI-01-01'}
matrix=[ (m[0],m[1],m[2],m[3],m[4],aimap.get(m[0],m[5]) if m[8]=='01' else m[5],*m[6:8]) for m in matrix]
def cell(s): return str(s).replace('|','\\|').replace('\n',' ')
a=['# AI 使用与核验日志','',
'由 T07 负责人从 `issues/issue-<编号>.md` 汇总。普通任务 PR 不编辑本文件。编号使用 `AI-<Issue号>-<序号>`，避免重复。原始提问、回答和核验过程见各 Issue 文件。','',
'| 编号 | 提问/PROMPT 摘要 | AI 结论 | 验证方法 | 结果 | 修正后的表述 | 记录人 |','|---|---|---|---|---|---|---|']
for r in ailog: a.append('| '+' | '.join(cell(fix(x)) for x in r)+' |')
a+=['','规则：AI 回答是待验证假设，而非引用证据。至少记录一个经实际代码运行证实“AI 错误或条件不充分”的案例。','',
'注：T07（#7）暂无 AI 记录；有新增后由 T07 负责人补充。','']
(ev/'ai-log.md').write_text('\n'.join(a))
m=['# 证据矩阵','',
'由 T07 负责人在各任务 PR 合并后汇总。普通任务 PR 只更新 `issues/issue-<编号>.md`，不编辑本文件。来源编号使用 `C-<Issue号>-<序号>`，例如 `C-05-01`，避免全组冲突。各行保留到原始证据的链接；“测试证据”列写“见该 Issue 文件”表示该文件未单列测试证据，汇总时未另行补充。','',
'| 结论编号 | 可检验结论 | 源码证据 | 测试证据 | 运行证据 | AI 记录 | 结论状态 | 负责人/复核人 |','|---|---|---|---|---|---|---|---|']
for r in matrix: m.append('| '+' | '.join(cell(x) or '—' for x in r[:8])+' |')
m+=['','状态只允许使用：`待验证`、`已支持`、`已否定`、`需限定`。','']
(ev/'evidence-matrix.md').write_text('\n'.join(m))
print(len(ailog),'ai rows',len(matrix),'matrix rows')
