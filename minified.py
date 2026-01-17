_R='iterations'
_Q='first'
_P='dropL'
_O='pickL'
_N='status'
_M='time'
_L='return'
_K='actions_done'
_J='actions'
_I=1.
_H='serveP'
_G='pos'
_F='load'
_E='parcels'
_D='ended'
_C=True
_B=None
_A=False
import sys,math,random,time,bisect
from typing import Any,Dict,List,Optional,Tuple,Union,Sequence,Iterator
Request=Tuple[int,int,str]
SwapRequest=Tuple[int,int,str]
Action=Tuple[int,str,int,int]
RelocateRequest=Tuple[int,int,int,int,str]
CostChange=Tuple[int,int,int]
RelocateAction=Tuple[RelocateRequest,CostChange]
class ShareARideProblem:
	def __init__(A,N,M,K,parcel_qty,vehicle_caps,dist,coords=_B,name=_B):A.N=N;A.M=M;A.K=K;A.q=list(parcel_qty);A.Q=list(vehicle_caps);A.D=[A[:]for A in dist];A.num_nodes=2*N+2*M+1;A.num_requests=N+M;A.num_actions=N+2*M+K;A.num_expansions=N+2*M;A.pserve=lambda pid:(pid,pid+N+M);A.lpick=lambda lid:N+lid;A.ldrop=lambda lid:2*N+M+lid;A.rev_ppick=lambda nodeid:nodeid;A.rev_pdrop=lambda nodeid:nodeid-(N+M);A.rev_lpick=lambda n:n-N;A.rev_ldrop=lambda n:n-(2*N+M);A.is_ppick=lambda nodeid:1<=nodeid<=N;A.is_pdrop=lambda nodeid:N+M+1<=nodeid<=2*N+M;A.is_lpick=lambda nodeid:N+1<=nodeid<=N+M;A.is_ldrop=lambda nodeid:2*N+M+1<=nodeid<=2*(N+M);A.coords=coords;A.name=name
	def is_valid(A):
		if len(A.q)!=A.M:return _A
		if len(A.Q)!=A.K:return _A
		if len(A.D)!=A.num_nodes:return _A
		if not all(len(B)==A.num_nodes for B in A.D):return _A
		return _C
	def copy(A):return ShareARideProblem(A.N,A.M,A.K,A.q[:],A.Q[:],[A[:]for A in A.D],A.coords,A.name)
def route_cost_from_sequence(seq,D,verbose=_A):
	A,B=0,0
	for C in seq[1:]:B+=D[A][C];A=C
	return B
class Solution:
	def __init__(A,problem,routes,route_costs=_B):
		E=route_costs;C=routes;B=problem
		if not C:raise ValueError('Routes list cannot be empty.')
		if len(C)!=B.K:raise ValueError(f"Expected {B.K} routes, got {len(C)}.")
		if not E:D=[route_cost_from_sequence(A,B.D)for A in C]
		else:D=E
		A.problem=B;A.routes=C;A.route_costs=D;A.num_actions=B.num_actions;A.max_cost=max(D)if D else 0
	def is_valid(G):
		A=G.problem;J=A.K
		if len(G.routes)!=J:return _A
		H=set()
		for(K,D)in enumerate(G.routes):
			if not(D[0]==0 and D[-1]==0):return _A
			L=len(D);E=0;F=set()
			for(I,B)in enumerate(D[1:-1],start=1):
				if B in H:return _A
				H.add(B)
				if A.is_ppick(B):
					M=A.rev_ppick(B);N=A.pserve(M)[1]
					if I+1>=L or D[I+1]!=N:return _A
				elif A.is_pdrop(B):0
				elif A.is_lpick(B):
					C=A.rev_lpick(B)
					if C in F:return _A
					E+=A.q[C-1]
					if E>A.Q[K]:return _A
					F.add(C)
				elif A.is_ldrop(B):
					C=A.rev_ldrop(B)
					if C not in F:return _A
					if E-A.q[C-1]<0:return _A
					E-=A.q[C-1];F.remove(C)
			if E!=0:return _A
		if len(H)!=A.num_requests*2:return _A
		return _C
	def stdin_print(A,verbose=_A):
		if verbose:print('Max cost:',A.max_cost);print()
		print(A.problem.K)
		for(B,C)in zip(A.routes,A.route_costs):print(len(B));print(' '.join(map(str,B)))
class PartialSolution:
	def __init__(A,problem,routes=_B):C=routes;B=problem;A.problem=B;A.routes=A._init_routes(C);A.route_costs=A._init_costs(C);A.min_cost=min(A.route_costs);A.max_cost=max(A.route_costs);A.avg_cost=sum(A.route_costs)/B.K;A.node_assignment=A._init_node_assignment();A.remaining_pass_serve,A.remaining_parc_pick,A.remaining_parc_drop,A.states,A.num_actions=A._init_states()
	def _init_routes(D,routes=_B):
		A=routes;B=D.problem.K
		if not A:return[[0]for A in range(B)]
		if len(A)!=B:raise ValueError(f"Expected {B} routes, got {len(A)}.")
		for C in A:
			if not C:raise ValueError('One route cannot be null')
			elif C[0]!=0:raise ValueError('Each route must start at depot 0.')
		return A
	def _init_costs(A,routes=_B):
		B=routes
		if not B:return[0]*A.problem.K
		if len(B)!=A.problem.K:raise ValueError('Mismatch between routes and route_costs length.')
		return[route_cost_from_sequence(B,A.problem.D)for B in B]
	def _init_node_assignment(B):
		C=len(B.problem.D);D=[-1]*C
		for(E,F)in enumerate(B.routes):
			for A in F[1:]:
				if A==0 or A>=C:continue
				D[A]=E
		return D
	def _init_states(I):
		A=I.problem;J=set(range(1,A.N+1));K=set(range(1,A.M+1));G=set();L=[];D=0
		for(T,E)in enumerate(I.routes):
			M=len(E);F=set();H=0
			for(N,C)in enumerate(E[1:],start=1):
				if A.is_ppick(C):
					O=A.rev_ppick(C);P=A.pserve(O)[1]
					if N+1>=M:raise RuntimeError('Invalid route: passenger pickup not followed by drop.')
					if E[N+1]!=P:raise RuntimeError('Invalid route: passenger pickup not followed by correct drop.')
					D+=1;J.discard(O)
				elif A.is_pdrop(C):0
				elif A.is_lpick(C):B=A.rev_lpick(C);F.add(B);H+=A.q[B-1];D+=1;K.discard(B);G.add(B)
				elif A.is_ldrop(C):
					B=A.rev_ldrop(C)
					if B not in F:raise RuntimeError('Invalid route: parcel drop without prior pickup.')
					F.remove(B);H-=A.q[B-1];D+=1;G.discard(B)
				else:
					if C!=0:raise RuntimeError('Invalid route: node id out of range.')
					D+=1
			Q=E[-1];R=M>1 and E[-1]==0;S={_G:Q,_E:F,_F:H,_J:D,_D:R};L.append(S)
		return J,K,G,L,D
	def is_valid(B,verbose=_A):
		A=B.problem;W,X,Q=A.N,A.M,A.K
		if not len(B.routes)==len(B.states)==len(B.route_costs)==Q:return _A
		if len(B.node_assignment)!=len(A.D):return _A
		R=set(range(1,W+1));S=set(range(1,X+1));L=set();M=[-1]*len(A.D);I=0;N=0;Y=0
		for F in range(Q):
			E=B.routes[F];J=len(E);O=B.states[F]
			if not E or E[0]!=0:return _A
			T=J>1 and E[-1]==0
			if O[_D]!=T:return _A
			G=set();J=len(E);H=0;U=E[0];K=0
			for(P,C)in enumerate(E[1:],start=1):
				if not 0<=C<A.num_nodes:return _A
				if C!=0:
					Z=M[C]
					if Z not in(-1,F):return _A
					M[C]=F
				K+=A.D[U][C];U=C
				if A.is_ppick(C):
					V=A.rev_ppick(C)
					if P+1<J:
						a=E[P+1];b=A.pserve(V)[1]
						if a!=b:return _A
					I+=1;R.discard(V)
				elif A.is_pdrop(C):0
				elif A.is_lpick(C):
					D=A.rev_lpick(C)
					if D in G:return _A
					H+=A.q[D-1]
					if H>A.Q[F]:return _A
					I+=1;G.add(D);S.discard(D);L.add(D)
				elif A.is_ldrop(C):
					D=A.rev_ldrop(C)
					if D not in G:return _A
					H-=A.q[D-1]
					if H<0:return _A
					I+=1;G.remove(D);L.discard(D)
				else:
					if P!=J-1:return _A
					if H!=0 or G:return _A
					if not T:return _A
					I+=1
			if O[_E]!=G:return _A
			if O[_F]!=H:return _A
			if B.route_costs[F]!=K:return _A
			N=max(N,K);Y+=K
		if R!=B.remaining_pass_serve:return _A
		if S!=B.remaining_parc_pick:return _A
		if L!=B.remaining_parc_drop:return _A
		if M!=B.node_assignment:return _A
		if B.max_cost!=N:return _A
		if B.num_actions!=I:return _A
		return _C
	def is_pending(A):return A.num_actions<A.problem.num_actions
	def is_identical(A,other):
		B=other
		if A is B:return _C
		if A.problem is not B.problem or A.num_actions!=B.num_actions:return _A
		return sorted(tuple(A[:3])for A in A.routes)==sorted(tuple(A[:3])for A in B.routes)
	def copy(A):return PartialSolution(problem=A.problem,routes=[A.copy()for A in A.routes]if A.routes else _B)
	def enumerate_action_nodes(B,route_idx):
		D=B.problem;E=B.routes[route_idx];C=[]
		for A in E:
			if D.is_ppick(A):F=B.problem.rev_ppick(A);G=B.problem.pserve(F)[1];C.append((A,G))
			else:C.append((A,A))
		return C
	def decrease_cost(A,route_idx,dec):B=route_idx;A.route_costs[B]-=dec;A.max_cost=max(A.route_costs);A.min_cost=min(A.min_cost,A.route_costs[B]);A.avg_cost=sum(A.route_costs)/A.problem.K
	def possible_expand(E,t_idx):
		H=t_idx;F=E.states[H]
		if F[_D]:return[]
		A=E.problem;G=F[_G];C=[]
		for I in E.remaining_pass_serve:J,K=A.pserve(I);D=A.D[G][J]+A.D[J][K];C.append((_H,I,D))
		for B in E.remaining_parc_pick:
			L=A.q[B-1]
			if F[_F]+L<=A.Q[H]:D=A.D[G][A.lpick(B)];C.append((_O,B,D))
		for B in F[_E]:D=A.D[G][A.ldrop(B)];C.append((_P,B,D))
		C.sort(key=lambda x:x[2]);return C
	def check_expand(A,route_idx,kind,actid):
		E=route_idx;C=actid;B=kind;D=A.states[E];F=A.problem
		if D[_D]:return _A
		if B==_H:return C in A.remaining_pass_serve
		if B==_O:return C in A.remaining_parc_pick and D[_F]+F.q[C-1]<=F.Q[E]
		if B==_P:return C in D[_E]
		raise ValueError(f"Unknown action kind: {B}")
	def check_return(B,route_idx):A=B.states[route_idx];return not(A[_D]or A[_E])
	def apply_extend(A,route_idx,kind,actid,inc):
		G=kind;C=actid;B=route_idx;I=A.routes[B];D=A.states[B];E=A.problem
		if D[_D]:raise ValueError(f"Cannot apply action on ended route {B}.")
		if G==_H:K,J=E.pserve(C);I.append(K);I.append(J);A.node_assignment[K]=B;A.node_assignment[J]=B;A.remaining_pass_serve.discard(C);D[_G]=J;D[_J]+=1;A.route_costs[B]+=inc;A.max_cost=max(A.max_cost,A.route_costs[B]);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions+=1;return
		elif G==_O:
			F=E.q[C-1]
			if D[_F]+F>E.Q[B]:raise ValueError(f"Taxi {B} capacity exceeded for parcel {C}.")
			H=E.lpick(C);D[_F]+=F;D[_E].add(C);A.remaining_parc_pick.discard(C);A.remaining_parc_drop.add(C)
		elif G==_P:
			F=E.q[C-1]
			if D[_F]-F<0:raise ValueError(f"Taxi {B} load cannot be negative after dropping parcel {C}.")
			H=E.ldrop(C);D[_F]-=F;D[_E].discard(C);A.remaining_parc_drop.discard(C)
		else:raise ValueError(f"Unknown action kind: {G}")
		D[_G]=H;D[_J]+=1;I.append(H);A.node_assignment[H]=B;A.route_costs[B]+=inc;A.max_cost=max(A.max_cost,A.route_costs[B]);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions+=1
	def apply_return(A,t_idx):
		C=t_idx;D=A.routes[C];B=A.states[C]
		if B[_D]:return
		if B[_E]:raise ValueError(f"Taxi {C} must drop all loads before returning to depot.")
		E=A.problem.D[B[_G]][0];D.append(0);B[_G]=0;B[_J]+=1;B[_D]=_C;A.route_costs[C]+=E;A.max_cost=max(A.max_cost,A.route_costs[C]);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions+=1
	def reverse_action(A,t_idx):
		G=t_idx;F=A.routes[G];C=A.states[G]
		if len(F)<=1:raise ValueError(f"No actions to reverse for taxi {G}.")
		B=A.problem;D=F[-1]
		if B.is_pdrop(D):
			J=F.pop();I=F.pop();L=B.rev_pdrop(J)
			if B.rev_ppick(I)!=L:raise ValueError('Inconsistent route state: pdrop not preceded by corresponding ppick.')
			H=F[-1];K=B.D[H][I]+B.D[I][J];C[_G]=H;C[_J]-=1;C[_D]=_A;A.remaining_pass_serve.add(L);A.node_assignment[J]=-1;A.node_assignment[I]=-1;A.route_costs[G]-=K;A.max_cost=max(A.route_costs);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions-=1;return
		D=F.pop();H=F[-1];K=B.D[H][D];C[_G]=H;C[_J]-=1;C[_D]=_A
		if B.is_lpick(D):E=B.rev_lpick(D);C[_F]-=B.q[E-1];C[_E].discard(E);A.remaining_parc_pick.add(E);A.remaining_parc_drop.discard(E)
		elif B.is_ldrop(D):E=B.rev_ldrop(D);C[_F]+=B.q[E-1];C[_E].add(E);A.remaining_parc_pick.discard(E);A.remaining_parc_drop.add(E)
		elif D==0:0
		else:raise ValueError(f"Unexpected node type to reverse: {D}")
		A.route_costs[G]-=K;A.max_cost=max(A.route_costs);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.node_assignment[D]=-1;A.num_actions-=1
	def is_completed(A,verbose=_A):
		if A.num_actions!=A.problem.num_actions:return _A
		if not all(A[_D]for A in A.states):return _A
		return _C
	def to_solution(A):
		if not A.is_completed(verbose=_C):return
		if not A.is_valid(verbose=_C):return
		B=Solution(problem=A.problem,routes=A.routes,route_costs=A.route_costs);return B
	@staticmethod
	def from_solution(sol):A=[A.copy()for A in sol.routes];return PartialSolution(problem=sol.problem,routes=A)
def weighted(kind,inc,pweight=.7):
	if kind==_H:return pweight*inc
	return inc+1
def action_weight(action,pweight=.7):A=action;return weighted(A[1],A[3],pweight)
def softmax_weighter(incs,t):
	A=incs;B,E=min(A),max(A);C=E-B
	if C<1e-06:return[_I]*len(A)
	D=[]
	for F in A:G=(F-B)/C;D.append((_I-G+.1)**(_I/t))
	return D
def balanced_scorer(partial,sample_size=8,w_std=.15,seed=_B):
	B=sample_size;A=partial;E=random.Random(seed);C=sorted(A.route_costs)
	if len(C)==1:return A.max_cost
	D=E.choices(C,k=B);F=sum(D)/B;G=sum((A-F)**2 for A in D)/B;H=G**.5;return A.max_cost+w_std*H
def check_general_action(partial,action):
	A=partial;B,C,D,E=action
	if C==_L:return A.check_return(B)
	return A.check_expand(B,C,D)
def apply_general_action(partial,action):
	A=partial;B,C,D,E=action
	if C==_L:A.apply_return(B)
	else:A.apply_extend(B,C,D,E)
def enumerate_actions_greedily(partial,width=_B,asymmetric=_C):
	D=width;A=partial
	if D is _B:D=10**9
	B=A.problem;I=[A for(A,B)in enumerate(A.states)if not B[_D]]
	if not I:return[]
	E=sorted(I,key=lambda idx:A.route_costs[idx]);J=len(E)
	if asymmetric:
		K=set();L=[]
		for C in E:
			M=tuple(A.routes[C])
			if M in K:continue
			K.add(M);L.append(C)
		E=L
	def T(aggressive):
		if not aggressive:return J
		return min(2 if B.K>=25 else 3 if B.K>=12 else 4 if B.K>=6 else 5,J)
	def U(aggressive):
		if not aggressive:return 10**9
		return min(2 if B.num_nodes>=500 else 4 if B.num_nodes>=200 else 6 if B.num_nodes>=100 else 8 if B.num_nodes>=50 else 12 if B.num_nodes>=25 else 16,D)
	def N(aggressive):
		F=aggressive;B=[];G=E
		if F:G=E[:T(aggressive=_C)]
		for H in G:
			C=A.possible_expand(H)
			if F:C=sorted(C,key=lambda item:item[2])[:U(aggressive=_C)]
			I=[(H,A,B,C)for(A,B,C)in C];B.extend(I);B.sort(key=lambda item:item[3]);B=B[:D]
		return B
	F=N(aggressive=_C)
	if not F:F=N(aggressive=_A)
	V=A.max_cost;G=[];H=[]
	for O in F:
		C,W,a,P=O;X=weighted(W,P);Q=X,O
		if A.route_costs[C]+P<=V:G.append(Q)
		else:H.append(Q)
	G.sort(key=lambda item:item[0]);H.sort(key=lambda item:item[0]);R=[A for(B,A)in G+H][:D]
	if not R:
		if A.num_actions<B.num_expansions:raise RuntimeError('Premature routes not covering all nodes.')
		S=[]
		for C in E:
			Y=A.states[C]
			if A.check_return(C):Z=B.D[Y[_G]][0];S.append((C,_L,0,Z))
		return S[:D]
	return R
def sample_from_weight(rng,weights):
	A=weights;C=sum(A)
	if C<1e-10:B=rng.randrange(len(A))
	else:
		E=rng.random()*C;D=.0;B=0
		for(F,G)in enumerate(A):
			D+=G
			if E<=D:B=F;break
	return B
def repair_one_route(partial,route_idx,steps,T=_I,seed=_B,verbose=_A):
	B=route_idx;A=partial;E=random.Random(seed);C=0
	for M in range(steps):
		F=A.states[B]
		if F[_D]:break
		D=A.possible_expand(B)
		if not D:A.apply_return(B);C+=1;break
		G=[weighted(A,B)for(A,C,B)in D];H=softmax_weighter(G,T);I=sample_from_weight(E,H);J,K,L=D[I];A.apply_extend(B,J,K,L);C+=1
	return A,C
def repair_operator(partial,repair_proba=_B,steps=_B,T=_I,seed=_B,verbose=_A):
	C=steps;B=repair_proba;A=partial;E=random.Random(seed)
	if B is _B:B=_I
	if C is _B:C=10**9
	I=list(range(A.problem.K));D=A.problem.K;J=round(B*D+.5);K=min(D,max(1,J));L=E.sample(I,K);F=0;G=[_A]*D
	for H in L:A,M=repair_one_route(partial=A,route_idx=H,steps=C,T=T,seed=E.randint(0,1000000),verbose=verbose);F+=M;G[H]=_C
	return A,G,F
def destroy_one_route(problem,route,route_idx,steps=10,verbose=_A):
	E='The destroyed route is likely invalid beforehand.';A=problem;C=route[:];D=0
	while D<steps and len(C)>1:
		B=C.pop()
		if A.is_pdrop(B):
			F=A.rev_pdrop(B);G=A.pserve(F)[0]
			if C.pop()!=G:raise RuntimeError(E)
		elif A.is_lpick(B)or A.is_ldrop(B)or B==0:0
		else:raise RuntimeError(E)
		D+=1
	return C,D
def destroy_operator(sol,destroy_proba,destroy_steps,seed=_B,t=_I,verbose=_A):
	A=sol;M=random.Random(seed);B=A.problem.K;D=[A[:]for A in A.routes];N=A.route_costs;O=round(destroy_proba*B+.5);P=min(B,max(1,O));Q=softmax_weighter(N,t=t);G=[];E=list(range(B));H=Q[:]
	for T in range(P):
		if not E:break
		F=sample_from_weight(M,H);G.append(E[F]);E.pop(F);H.pop(F)
	I=[_A]*B;J=0
	for C in G:
		K=D[C]
		if len(K)<=2:continue
		R,L=destroy_one_route(A.problem,K,C,steps=destroy_steps,verbose=verbose)
		if L>0:D[C]=R;I[C]=_C;J+=L
	S=PartialSolution(problem=A.problem,routes=D);return S,I,J
class MinMaxPfsumArray:
	class Block:
		def __init__(A,data):A.arr=data[:];A.size=len(A.arr);A.recalc()
		def recalc(A):
			A.size=len(A.arr);A.sum=sum(A.arr);B=0;C=10**18;D=-10**18
			for E in A.arr:B+=E;C=min(C,B);D=max(D,B)
			A.min_pref=C;A.max_pref=D
		def insert(A,idx,entry):A.arr.insert(idx,entry);A.recalc()
		def erase(A,idx):del A.arr[idx];A.recalc()
	def __init__(A,data):A.block_arr=[];A.num_data=0;A.block_prefix=[];A.build(data)
	def build(A,data):
		A.block_arr.clear();A.num_data=len(data);A.block_size=max(0,int(math.sqrt(A.num_data)))+2
		for B in range(0,A.num_data,A.block_size):A.block_arr.append(A.Block(data[B:B+A.block_size]))
		A.num_block=len(A.block_arr);A._rebuild_indexing()
	def _rebuild_indexing(A):
		A.block_prefix=[];B=0
		for C in A.block_arr:A.block_prefix.append(B);B+=C.size
		A.num_data=B
	def _find_block(A,idx):
		B=idx
		if B>A.num_data:B=A.num_data
		C=bisect.bisect_right(A.block_prefix,B)-1;D=B-A.block_prefix[C];return C,D
	def insert(A,idx,val):
		B=val
		if idx==A.num_data:
			if not A.block_arr:A.block_arr.append(A.Block([B]))
			else:
				C=A.block_arr[-1]
				if C.size>=2*A.block_size:A.block_arr.append(A.Block([B]))
				else:C.insert(C.size,B)
			A.num_data+=1;A._rebuild_indexing();return
		D,H=A._find_block(idx);E=A.block_arr[D];E.insert(H,B)
		if E.size>2*A.block_size:F=E.arr;G=len(F)//2;I=A.Block(F[:G]);J=A.Block(F[G:]);A.block_arr[D:D+1]=[I,J]
		A.num_data+=1;A._rebuild_indexing()
	def delete(A,idx):
		B,F=A._find_block(idx);A.block_arr[B].erase(F)
		if A.block_arr[B].size==0:del A.block_arr[B]
		else:
			G=max(1,A.block_size//2)
			if A.block_arr[B].size<G:
				if B+1<len(A.block_arr):
					D=A.block_arr[B+1]
					if A.block_arr[B].size+D.size<=2*A.block_size:C=A.block_arr[B].arr+D.arr;A.block_arr[B:B+2]=[A.Block(C)]
				elif B-1>=0:
					E=A.block_arr[B-1]
					if E.size+A.block_arr[B].size<=2*A.block_size:C=E.arr+A.block_arr[B].arr;A.block_arr[B-1:B+1]=[A.Block(C)]
		A.num_data-=1;A._rebuild_indexing()
	def query_min_prefix(I,l,r):
		B=10**18;A=0;C=0;B=10**18;A=0
		for D in I.block_arr:
			E=D.size
			if A+E<=l:C+=D.sum;A+=E;continue
			if A>=r:break
			F=max(0,l-A);H=min(E,r-A)
			if F>0:
				for G in range(F):C+=D.arr[G]
			if F==0 and H==E:B=min(B,C+D.min_pref);C+=D.sum
			else:
				for G in range(F,H):C+=D.arr[G];B=min(B,C)
			A+=E
		return B
	def query_max_prefix(J,l,r):
		I='-inf';B=float(I);A=0;C=0;B=float(I);A=0
		for D in J.block_arr:
			E=D.size
			if A+E<=l:C+=D.sum;A+=E;continue
			if A>=r:break
			F=max(0,l-A);H=min(E,r-A)
			if F>0:
				for G in range(F):C+=D.arr[G]
			if F==0 and H==E:B=max(B,C+D.max_pref);C+=D.sum
			else:
				for G in range(F,H):C+=D.arr[G];B=max(B,C)
			A+=E
		return B
	def get_data_point(A,idx):
		B=idx
		if B<0 or B>=A.num_data:raise IndexError('Index out of bounds')
		C,D=A._find_block(B);return A.block_arr[C].arr[D]
	def get_data_segment(C,l,r):
		if l<0 or r<0 or l>r or r>C.num_data:raise IndexError('Invalid segment range')
		D=[];A=0
		for E in C.block_arr:
			B=E.size
			if A>=r:break
			if A+B<=l:A+=B;continue
			F=max(0,l-A);G=min(B,r-A)
			if G>F:D.extend(E.arr[F:G])
			A+=B
		return D
	def get_data(A):return A.get_data_segment(0,A.num_data)
def cost_decrement_relocate(partial,rfidx,rtidx,pfidx,qfidx,ptidx,qtidx):
	R=qtidx;Q=ptidx;P=rtidx;O=rfidx;I=qfidx;H=pfidx;B=partial;A=B.problem.D;Y=B.max_cost;E=B.routes[O];J=B.routes[P];Z=B.route_costs[O];a=B.route_costs[P];C=E[H];D=E[I];K=E[H-1];S=E[H+1];T=E[I-1];L=E[I+1];F=0
	if H+1==I:F-=A[K][C]+A[C][D]+A[D][L];F+=A[K][L]
	else:F-=A[K][C]+A[C][S]+A[T][D]+A[D][L];F+=A[K][S]+A[T][L]
	U=Z+F;M=J[Q-1];V=J[Q];W=J[R-2];N=J[R-1];G=0
	if R==Q+1:G-=A[M][N];G+=A[M][C]+A[C][D]+A[D][N]
	else:G-=A[M][V]+A[W][N];G+=A[M][C]+A[C][V]+A[W][D]+A[D][N]
	X=a+G;b=[B.route_costs[A]for A in range(B.problem.K)if A!=O and A!=P];c=max(U,X,*b);return U,X,Y-c
def relocate_from_to(partial,route_from_idx,route_to_idx,steps,mode,uplift=1,seed=_B,verbose=_A):
	N='serveL';L=mode;K=partial;F=route_from_idx;E=route_to_idx;O=random.Random(seed);A=K.problem;C=K.copy();B=C.routes[F];D=C.routes[E];H=len(B);G=len(D)
	if H<5:return K,[_A]*A.K,0
	def M(route,n):
		E=[0]*n
		for(F,B)in enumerate(route):
			if A.is_lpick(B):C=A.rev_lpick(B);D=A.q[C-1]
			elif A.is_ldrop(B):C=A.rev_ldrop(B);D=-A.q[C-1]
			else:D=0
			E[F]=D
		G=MinMaxPfsumArray(E);return G
	I=M(B,H);J=M(D,G);P=A.Q[F];Q=A.Q[E]
	def R(req):
		D,E,A,B,C=req
		if C==N:return _C
		return B==A+1
	def S(req):
		D,E,F,G,K=req
		if K==_H:return _C
		H=B[D]
		if A.is_lpick(H):L=A.rev_lpick(H);C=A.q[L-1]
		else:C=0
		M=I.query_min_prefix(D,E);N=I.query_max_prefix(D,E)
		if M-C<0:return _A
		if N-C>P:return _A
		O=J.query_min_prefix(F-1,G-1);R=J.query_max_prefix(F-1,G-1)
		if O+C<0:return _A
		if R+C>Q:return _A
		return _C
	def T(req):
		A=req
		if not R(A):return
		if not S(A):return
		B=cost_decrement_relocate(C,F,E,A[0],A[1],A[2],A[3]);return B
	def U():
		M={B:A for(A,B)in enumerate(B)};H=[]
		for(E,I)in enumerate(B[1:],start=1):
			if A.is_ppick(I):C=E+1;H.append((E,C,_H))
			elif A.is_lpick(I):
				O=A.rev_lpick(I);P=A.ldrop(O);C=M.get(P)
				if C is not _B and C>E:H.append((E,C,N))
		Q=[(B,B+1)for B in range(1,G)if not A.is_ppick(D[B-1])];R=[(B,C)for B in range(1,G)if not A.is_ppick(D[B-1])for C in range(B+1,G+1)if not A.is_ppick(D[C-2])]
		for(E,C,K)in H:
			S=Q if K==_H else R
			for(U,V)in S:
				J=E,C,U,V,K;F=T(J)
				if F is _B:continue
				W,W,X=F
				if X<uplift:continue
				if L==_Q:yield(J,F);return
				else:yield(J,F)
	def V():
		A=list(U())
		if not A:return
		if L=='stochastic':return O.choice(A)
		elif L=='best':return max(A,key=lambda x:x[1][2])
		else:return A[0]
	def W(action):nonlocal B,D,C;(A,G,J,K,O),(L,M,N)=action;H=B[A];I=B[G];del B[G];del B[A];D.insert(J,H);D.insert(K,I);C.routes[F]=B;C.routes[E]=D;C.route_costs[F]=L;C.route_costs[E]=M;C.max_cost-=N;C.node_assignment[H]=E;C.node_assignment[I]=E
	def X(action):
		nonlocal I,J,B,D;(C,E,G,H,M),N=action;K=B[C];L=B[E]
		def F(nodeid):
			B=nodeid
			if A.is_lpick(B):C=A.rev_lpick(B);return A.q[C-1]
			elif A.is_ldrop(B):C=A.rev_ldrop(B);return-A.q[C-1]
			else:return 0
		I.delete(E);I.delete(C);J.insert(G,F(K));J.insert(H,F(L))
	def Y():
		nonlocal H,G,B,D;C=0;I=[_A]*A.K
		while C<steps:
			J=V()
			if J is _B:break
			X(J);W(J);C+=1;I[F]=_C;I[E]=_C;H-=2;G+=2
			if H<5:break
		return I,C
	Z,a=Y();return C,Z,a
def relocate_operator(partial,steps=_B,mode=_Q,uplift=1,seed=_B,verbose=_A):
	E=partial;B=steps;C=E.problem.K
	if C<2:return E.copy(),[_A]*C,0
	if B==_B:B=10**9
	M=random.Random(seed);A=E.copy();H=[_A]*C;D=0
	while D<B:
		I=list(enumerate(A.route_costs));F=max(I,key=lambda x:x[1])[0];N=[A for(A,B)in sorted(I,key=lambda x:x[1])]
		if len(A.routes[F])<5:break
		J=_A
		for G in N:
			if G==F:continue
			if len(A.routes[G])<2:continue
			O=B-D;P,Q,K=relocate_from_to(A,route_from_idx=F,route_to_idx=G,steps=O,mode=mode,uplift=uplift,seed=M.randint(10,10**9),verbose=verbose)
			if K>0:
				A=P;D+=K
				for L in range(C):
					if Q[L]:H[L]=_C
				J=_C;break
		if not J:break
	return A,H,D
def greedy_solver(problem,partial=_B,num_actions=7,t_actions=.01,seed=_B,verbose=_A):
	A=partial;D=time.time();G=random.Random(seed)
	if A is _B:A=PartialSolution(problem=problem)
	C=0;E=A.num_actions
	while A.is_pending():
		C+=1;F=enumerate_actions_greedily(A,num_actions);B=[A for A in F if A[1]!=_L]
		if not B:B=F
		if not B:return _B,{_R:C,_M:time.time()-D,_K:A.num_actions-E,_N:'error'}
		H=[A[3]for A in B];I=softmax_weighter(H,t_actions);J=sample_from_weight(G,I);K=B[J];apply_general_action(A,K)
	L=A.to_solution();M={_R:C,_M:time.time()-D,_K:A.num_actions-E,_N:'done'};return L,M
def iterative_greedy_solver(problem,partial=_B,iterations=10000,num_actions=7,t_actions=.01,destroy_proba=.53,destroy_steps=13,destroy_t=1.3,rebuild_proba=.29,rebuild_steps=3,rebuild_t=1.2,time_limit=3e1,seed=_B,verbose=_A):
	N=rebuild_steps;M=rebuild_proba;L=destroy_steps;K=destroy_proba;G=partial;F=problem;B=seed;H=time.time();O=H+time_limit;X=random.Random(B);assert 1e-05<K<.99999;assert 1e-05<M<.99999;assert 1<=N<=L
	if G is _B:G=PartialSolution(problem=F,routes=[])
	A,Y=greedy_solver(problem=F,partial=G,num_actions=num_actions,t_actions=t_actions,seed=3*B if B else _B,verbose=verbose)
	if not A:return _B,{_M:time.time()-H,_N:'error'}
	D=A.max_cost;E=Y[_K];P=D;Q=0;R=0;S=0;T='done';I=0
	for Z in range(1,iterations+1):
		if O and time.time()>=O:T='overtime';break
		U=_B if B is _B else 2*B+98*Z;a,b,V=destroy_operator(A,K,L,seed=U,t=destroy_t);R+=V;E+=V;J=a
		for(c,d)in enumerate(b):
			if not d:continue
			if X.random()>M:continue
			J,W=repair_one_route(J,route_idx=c,steps=N,T=rebuild_t,seed=U);S+=W;E+=W
		C,e=greedy_solver(F,partial=J,num_actions=1,verbose=_A);P+=C.max_cost if C else 0;E+=e[_K];I+=1 if C else 0
		if C and C.max_cost<D:A=C;D=C.max_cost;Q+=1
	f=PartialSolution.from_solution(A);g,h,h=relocate_operator(f,mode=_Q,seed=_B if B is _B else 4*B+123);A=g.to_solution();assert A;D=A.max_cost;i=time.time()-H;j={_R:I,_K:E,'improvements':Q,'actions_destroyed':R,'actions_rebuilt':S,'average_cost':P/(I+1),_M:i,_N:T};return A,j
def read_instance():
	A,B,D=map(int,sys.stdin.readline().strip().split());E=list(map(int,sys.stdin.readline().split()));F=list(map(int,sys.stdin.readline().split()));C=[[0]*(2*A+2*B+1)for C in range(2*A+2*B+1)]
	for G in range(2*A+2*B+1):H=sys.stdin.readline().strip();C[G]=list(map(int,H.split()))
	return ShareARideProblem(A,B,D,E,F,C)
def main(verbose=_A):
	H=verbose;G=read_instance();A=G.num_nodes;A=G.num_nodes
	if A<=100:B,C,D,E,F=9,3,9000,6,.6
	elif A<=250:B,C,D,E,F=8,2,4000,4,.5
	elif A<=500:B,C,D,E,F=7,1,1400,3,.3
	elif A<=1000:B,C,D,E,F=6,0,600,2,.1
	else:B,C,D,E,F=5,0,250,2,.05
	I,J=iterative_greedy_solver(problem=G,iterations=25000,time_limit=25e1,verbose=H,seed=42);assert I;I.stdin_print(verbose=H)
if __name__=='__main__':main(verbose=_A)