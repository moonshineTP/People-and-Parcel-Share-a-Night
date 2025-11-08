_U='stochastic'
_T='Cannot swap depot nodes.'
_S='Index out of bounds'
_R='best'
_Q='-inf'
_P='time'
_O='iterations'
_N='load'
_M='route'
_L='parcels'
_K='pos'
_J='first'
_I='inf'
_H='actions_evaluated'
_G='cost'
_F='passenger'
_E=1.
_D='ended'
_C=True
_B=None
_A=False
import sys,time,time,random
from typing import Any,List,Optional,Tuple,Dict
class ShareARideProblem:
	def __init__(A,N,M,K,parcel_qty,vehicle_caps,dist,coords=_B):A.N=N;A.M=M;A.K=K;A.q=list(parcel_qty);A.Q=list(vehicle_caps);A.D=[A[:]for A in dist];A.num_nodes=2*N+2*M+1;A.num_requests=N+M;A.ppick=lambda i:i;A.pdrop=lambda i:N+M+i;A.parc_pick=lambda j:N+j;A.parc_drop=lambda j:2*N+M+j;A.rev_ppick=lambda i:i;A.rev_pdrop=lambda n:n-(N+M);A.rev_parc_pick=lambda n:n-N;A.rev_parc_drop=lambda n:n-(2*N+M);A.is_ppick=lambda x:1<=x<=N;A.is_pdrop=lambda x:N+M+1<=x<=2*N+M;A.is_parc_pick=lambda x:N+1<=x<=N+M;A.is_parc_drop=lambda x:2*N+M+1<=x<=2*(N+M);A.coords=coords
	def is_valid(A):
		try:assert len(A.q)==A.M;assert len(A.Q)==A.K;assert len(A.D)==A.num_nodes;assert all(len(B)==A.num_nodes for B in A.D);assert len(A.coords)==A.num_nodes if A.coords is not _B else _C;return _C
		except:return _A
	def copy(A):return ShareARideProblem(A.N,A.M,A.K,list(A.q),list(A.Q),[A[:]for A in A.D])
	def stdin_print(A):
		print(A.N,A.M,A.K);print(*A.q);print(*A.Q)
		for B in A.D:print(*B)
def route_cost_from_sequence(seq,D,verbose=_A):
	E=verbose;A=seq;assert A and A[0]==0;B,F=0,0
	for C in A[1:]:
		if E:print(D[B][C],end=' ')
		F+=D[B][C];B=C
	if E:print()
	return F
class Solution:
	def __init__(B,problem,routes,route_costs=_B):
		E=route_costs;C=problem;A=routes
		if not A:raise ValueError('Routes list cannot be empty.')
		if len(A)!=C.K:raise ValueError(f"Expected {C.K} routes, got {len(A)}.")
		if not E:D=[route_cost_from_sequence(A,C.D)for A in A]
		else:D=E
		B.problem=C;B.routes=A;B.route_costs=D;B.max_cost=max(D)if D else 0
	def is_valid(G):
		A=G.problem;M,N,K=A.N,A.M,A.K
		if len(G.routes)!=K:return _A
		for(L,H)in enumerate(G.routes):
			if not(H[0]==0 and H[-1]==0):return _A
			D=set();F=set();E=0;I=set();J=set()
			for C in H[1:-1]:
				if A.is_ppick(C):
					id=A.rev_ppick(C)
					if id in I:return _A
					if len(D)>=1:return _A
					D.add(id);I.add(id)
				elif A.is_pdrop(C):
					id=A.rev_pdrop(C)
					if id not in D:return _A
					D.remove(id)
				elif A.is_parc_pick(C):
					B=A.rev_parc_pick(C)
					if B in J or B in F:return _A
					E+=A.q[B-1]
					if E>A.Q[L]:return _A
					J.add(B);F.add(B)
				elif A.is_parc_drop(C):
					B=A.rev_parc_drop(C)
					if B not in F:return _A
					assert E-A.q[B-1]>=0;E-=A.q[B-1];F.remove(B)
			if D:return _A
			if E!=0:return _A
		return _C
	def stdin_print(A,verbose=0):
		B=verbose
		if B:print(f"*** Max route cost: {A.max_cost} ***")
		print(A.problem.K);assert len(A.routes)==len(A.route_costs)
		for(C,D)in zip(A.routes,A.route_costs):
			if B:print(f"- Route cost: {D}")
			print(len(C));print(' '.join(map(str,C)))
class PartialSolution:
	def __init__(A,problem,routes=[]):B=routes;A.problem=problem;A.routes=A._init_routes(B);A.route_costs=A._init_costs(B);A.max_cost=max(A.route_costs);A.node_assignment=A._init_node_assignment();A.remaining_pass_pick,A.remaining_pass_drop,A.remaining_parc_pick,A.remaining_parc_drop,A.route_states=A._init_states()
	def _init_routes(D,routes):
		A=routes;B=D.problem.K
		if not A:return[[0]for A in range(B)]
		if len(A)!=B:raise ValueError(f"Expected {B} routes, got {len(A)}.")
		for C in A:
			if not C:raise ValueError('One route cannot be null')
			if C[0]!=0:raise ValueError('Each route must start at depot 0.')
		return A
	def _init_costs(A,routes):
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
	def _init_states(H):
		A=H.problem;L=set(range(1,A.N+1));I=set();M=set(range(1,A.M+1));J=set();N=[]
		for(O,E)in enumerate(H.routes):
			F=0;G=set();K=0
			for C in E[1:]:
				if A.is_ppick(C):D=A.rev_ppick(C);L.discard(D);I.add(D);F=D
				elif A.is_pdrop(C):
					D=A.rev_pdrop(C);I.discard(D)
					if F==D:F=0
				elif A.is_parc_pick(C):B=A.rev_parc_pick(C);M.discard(B);J.add(B);G.add(B);K+=A.q[B-1]
				elif A.is_parc_drop(C):
					B=A.rev_parc_drop(C)
					if B in G:G.remove(B);K-=A.q[B-1]
					J.discard(B)
			P=E[-1];Q=len(E)>1 and E[-1]==0;R={_M:E,_K:P,_G:H.route_costs[O],_N:K,_F:F,_L:set(G),_D:Q};N.append(R)
		return L,I,M,J,N
	def copy(A):return PartialSolution(problem=A.problem,routes=[list(A)for A in A.routes])
	def possible_actions(G,t_idx):
		I=t_idx;C=G.route_states[I]
		if C[_D]:return[]
		A=G.problem;H=C[_K];E=[]
		if C[_F]==0:
			for F in list(G.remaining_pass_pick):B=A.D[H][A.ppick(F)];E.append(('pickP',F,B))
		else:F=C[_F];B=A.D[H][A.pdrop(F)];E.append(('dropP',F,B))
		for D in list(G.remaining_parc_pick):
			J=A.q[D-1]
			if C[_N]+J<=A.Q[I]:B=A.D[H][A.parc_pick(D)];E.append(('pickL',D,B))
		for D in list(C[_L]):B=A.D[H][A.parc_drop(D)];E.append(('dropL',D,B))
		return E
	def apply_action(C,t_idx,kind,node_idx,inc):
		G=kind;D=t_idx;A=node_idx;B=C.route_states[D]
		if B[_D]:raise ValueError(f"Cannot apply action on ended route {D}.")
		E=C.problem
		if G=='pickP':
			if B[_F]!=0:raise ValueError(f"Taxi {D} already has passenger {B[_F]}.")
			F=E.ppick(A);B[_F]=A;C.remaining_pass_pick.discard(A);C.remaining_pass_drop.add(A)
		elif G=='dropP':
			if B[_F]!=A:raise ValueError(f"Taxi {D} is not carrying passenger {A}.")
			F=E.pdrop(A);B[_F]=0;C.remaining_pass_drop.discard(A)
		elif G=='pickL':
			H=E.q[A-1]
			if B[_N]+H>E.Q[D]:raise ValueError(f"Taxi {D} capacity exceeded for parcel {A}.")
			F=E.parc_pick(A);B[_N]+=H;B[_L].add(A);C.remaining_parc_pick.discard(A);C.remaining_parc_drop.add(A)
		elif G=='dropL':
			if A not in B[_L]:raise ValueError(f"Taxi {D} does not carry parcel {A}.")
			F=E.parc_drop(A);B[_N]-=E.q[A-1];B[_L].discard(A);C.remaining_parc_drop.discard(A)
		else:raise ValueError(f"Unknown action kind: {G}")
		B[_M].append(F);B[_G]+=inc;B[_K]=F;C.node_assignment[F]=D;C.route_costs[D]=B[_G];C.max_cost=max(C.max_cost,B[_G])
	def apply_return_to_depot(B,t_idx):
		C=t_idx;A=B.route_states[C]
		if A[_D]:return
		if A[_K]==0 and len(A[_M])>1:A[_D]=_C;return
		if A[_F]!=0 or A[_L]:raise ValueError(f"Taxi {C} must drop all loads before returning to depot.")
		A[_G]+=B.problem.D[A[_K]][0];B.route_costs[C]=A[_G];B.max_cost=max(B.max_cost,A[_G]);A[_M].append(0);A[_K]=0;A[_D]=_C
	def is_complete(A):return all(A[_D]for A in A.route_states)
	def to_solution(A):
		if not A.is_complete():print('Cannot convert to Solution: not all routes have ended at depot.');return
		B=Solution(problem=A.problem,routes=A.routes,route_costs=A.route_costs)
		if B is _B or not B.is_valid():print('Warning: Converted solution is not valid.')
		return B
	@staticmethod
	def from_solution(sol):A=[list(A)for A in sol.routes];return PartialSolution(problem=sol.problem,routes=A)
from typing import List,Tuple,Optional
def sample_from_weight(rng,weights):
	A=weights;C=sum(A)
	if C<1e-10:B=rng.randrange(len(A))
	else:
		E=rng.random()*C;D=.0;B=0
		for(F,G)in enumerate(A):
			D+=G
			if E<=D:B=F;break
	return B
def softmax_weighter(incs,T):
	A=incs;B,E=min(A),max(A);C=E-B
	if C<1e-06:return[_E]*len(A)
	D=[]
	for F in A:G=(F-B)/C;D.append((_E-G+.1)**(_E/T))
	return D
def repair_operator(partial,route_idx,steps=5,T=_E,seed=42,verbose=_A):
	F=steps;D=verbose;B=partial;A=route_idx;assert F>0,'Number of steps must be positive.';assert T>1e-05,'Temperature T must be positive.';H=random.Random(seed);E=0
	for P in range(F):
		I=B.route_states[A]
		if I[_D]:break
		C=B.possible_actions(A)
		if D:print(f"[build] route {A} available actions: {C}")
		if not C:
			if D:print(f"[build] route {A} has no feasible actions, ending.");B.apply_return_to_depot(A);E+=1;break
		J=[A[2]for A in C];K=softmax_weighter(J,T);G=sample_from_weight(H,K);L,M,N=C[G]
		if D:print(f"[build] route {A} selected action: {C[G]}")
		B.apply_action(A,L,M,N);E+=1
	if D:print(f"[build] route {A} finished building, added {E} nodes.")
	O=[B==A for B in range(B.problem.K)];return B,O,E
def destroy_one_route(route,route_idx,steps=10,verbose=_A):
	D=route;A=D[:-1];B=min(steps,max(0,len(A)-1))
	if B<=0:return D[:]
	E=len(A)-B;C=A[:E]
	if not C:C=[0]
	if verbose:print(f"[Operator: Destroy]: last {B} nodes from route {route_idx} removed.")
	return C
def destroy_operator(sol,destroy_proba,destroy_steps,seed=42,T=_E):
	A=sol;M=random.Random(seed);B=[A[:]for A in A.routes];C=A.route_costs;G=[_A]*len(B);H=0
	if not B:return PartialSolution(problem=A.problem,routes=B),G,H
	S=round(destroy_proba*len(B)+.5);I=min(A.problem.K,max(1,S));N=min(C)if C else .0;U=max(C)if C else _E;O=U-N;V=max(T,1e-06)
	if O<1e-06:D=M.sample(range(A.problem.K),I)
	else:
		P=[]
		for W in C:X=(W-N)/O;P.append((X+.1)**(_E/V))
		D=[];E=list(range(A.problem.K));J=P
		for a in range(I):
			Y=sum(J)
			if Y<1e-10:D.extend(E[:I-len(D)]);break
			else:
				K=sample_from_weight(M,J);D.append(E[K]);E.pop(K);J.pop(K)
				if not E:break
	for F in D:
		L=B[F]
		if len(L)<=2:continue
		Q=destroy_one_route(L,F,steps=destroy_steps,verbose=_A);R=max(0,len(L)-len(Q))
		if R>0:B[F]=Q;G[F]=_C;H+=R
	Z=PartialSolution(problem=A.problem,routes=B);return Z,G,H
def greedy_balanced_solver(prob,premature_routes=[],verbose=_A):
	G=verbose;I=time.time();A=PartialSolution(problem=prob,routes=premature_routes);D=A.route_states
	def J():return bool(A.remaining_pass_pick or A.remaining_pass_drop or A.remaining_parc_pick or A.remaining_parc_drop)
	E={_O:0,_H:0}
	while J():
		E[_O]+=1;H=[A for(A,B)in enumerate(D)if not B[_D]]
		if not H:break
		B=min(H,key=lambda i:D[i][_G]);F=A.possible_actions(B);E[_H]+=len(F)
		if G:print(f"Taxi with min cost: {B}");print(f"Actions available: {F}")
		if not F:A.apply_return_to_depot(B);continue
		K,L,M=min(F,key=lambda x:x[2]);A.apply_action(B,K,L,M)
		if G:print(f"Taxi: {B}: {D[B][_M]}\n")
	for(N,O)in enumerate(D):
		if not O[_D]:A.apply_return_to_depot(N)
	if G:print('All tasks completed.')
	C=A.to_solution();P=time.time()-I;Q={_O:E[_O],_H:E[_H],_P:P}
	if C and not C.is_valid():C=_B
	assert C.is_valid()if C else _C;return C,Q
def iterative_greedy_balanced_solver(prob,iterations=10,time_limit=1e1,seed=42,verbose=_A,destroy_proba=.4,destroy_steps=15,destroy_T=_E,rebuild_proba=.3,rebuild_steps=5,rebuild_T=_E):
	V='status';M=rebuild_steps;L=rebuild_proba;K=destroy_steps;J=destroy_proba;I=verbose;H=time_limit;assert 1e-05<J<.99999;assert 1e-05<L<.99999;assert 1<=M<=K;W=random.Random(seed);E=time.time();N=E+H if H is not _B else _B;A,X=greedy_balanced_solver(prob,verbose=_A)
	if not A:return _B,{_P:time.time()-E,V:'error'}
	C=A.max_cost;O=X[_H];P=0;Q=0;R=0;S='done';T=0
	if I:print(f"[iter 0] initial best cost: {C}")
	for U in range(1,iterations+1):
		if N and time.time()>=N:S='timeout';break
		T+=1;F=2*seed+U;D,Y,Z=destroy_operator(A,J,K,seed=F,T=destroy_T);Q+=Z
		for(G,a)in enumerate(Y):
			if not a or len(D.routes[G])<=2:continue
			if W.random()>L:continue
			D,f,b=repair_operator(D,route_idx=G,steps=M,T=rebuild_T,seed=F+G if F is not _B else _B,verbose=_A);R+=b
		B,c=greedy_balanced_solver(prob,premature_routes=D.routes,verbose=_A);O+=c[_H]
		if B and B.is_valid()and B.max_cost<C:
			A=B;C=B.max_cost;P+=1
			if I:print(f"[iter {U}] improved best to {C}")
	d=time.time()-E;e={_O:T,'improvements':P,_H:O,'nodes_destroyed':Q,'nodes_rebuilt':R,_P:d,V:S};return A,e
import heapq
from typing import List,Optional,Tuple,Callable,Union,Sequence
import math,bisect
class TreeSegment:
	def __init__(A,data,op,identity,sum_like=_C,add_neutral=0):
		A.n_elements=len(data);A.op=op;A.identity=identity;A.sum_like=sum_like;A.n_leaves=1
		while A.n_leaves<A.n_elements:A.n_leaves*=2
		A.data=[A.identity]*(2*A.n_leaves);A.lazy=[add_neutral]*(2*A.n_leaves)
		for B in range(A.n_elements):A.data[A.n_leaves+B]=data[B]
		for B in range(A.n_leaves-1,0,-1):A.data[B]=A.op(A.data[2*B],A.data[2*B+1])
	def _apply(A,x,val,length):
		B=val
		if A.sum_like:A.data[x]+=B*length
		else:A.data[x]+=B
		if x<A.n_leaves:A.lazy[x]+=B
	def _push(A,x,length):
		B=length
		if A.lazy[x]!=0:A._apply(2*x,A.lazy[x],B//2);A._apply(2*x+1,A.lazy[x],B//2);A.lazy[x]=0
	def _update(A,l,r,val,x,lx,rx):
		D=val;C=rx;B=lx
		if B>=r or C<=l:return
		if B>=l and C<=r:A._apply(x,D,C-B);return
		A._push(x,C-B);E=(B+C)//2;A._update(l,r,D,2*x,B,E);A._update(l,r,D,2*x+1,E,C);A.data[x]=A.op(A.data[2*x],A.data[2*x+1])
	def _query(A,l,r,x,lx,rx):
		C=rx;B=lx
		if B>=r or C<=l:return A.identity
		if B>=l and C<=r:return A.data[x]
		A._push(x,C-B);D=(B+C)//2;E=A._query(l,r,2*x,B,D);F=A._query(l,r,2*x+1,D,C);return A.op(E,F)
	def update(A,l,r,val):A._update(l,r,val,1,0,A.n_leaves)
	def query(A,l,r):return A._query(l,r,1,0,A.n_leaves)
class MinMaxPfsumArray:
	class Block:
		def __init__(A,data):A.arr=data[:];A.size=len(A.arr);A.recalc()
		def recalc(A):
			A.size=len(A.arr);A.sum=sum(A.arr);B=0;C=float(_I);D=float(_Q)
			for E in A.arr:B+=E;C=min(C,B);D=max(D,B)
			A.min_pref=C;A.max_pref=D
		def insert(A,idx,entry):A.arr.insert(idx,entry);A.recalc()
		def erase(A,idx):del A.arr[idx];A.recalc()
	def __init__(A,data):assert data;A.block_arr=[];A.n_data=0;A.block_prefix=[];A.build(data)
	def build(A,data):
		A.block_arr.clear();A.n_data=len(data);A.block_size=max(0,int(math.sqrt(A.n_data)))+2
		for B in range(0,A.n_data,A.block_size):A.block_arr.append(A.Block(data[B:B+A.block_size]))
		A.n_block=len(A.block_arr);A._rebuild_indexing()
	def _rebuild_indexing(A):
		A.block_prefix=[];B=0
		for C in A.block_arr:A.block_prefix.append(B);B+=C.size
		A.n_data=B
	def _find_block(A,idx):
		B=idx;assert A.block_arr,'No blocks present';assert 0<=B<A.n_data,_S
		if B>A.n_data:B=A.n_data
		C=bisect.bisect_right(A.block_prefix,B)-1;D=B-A.block_prefix[C];return C,D
	def insert(A,idx,val):B,C=A._find_block(idx);A.block_arr[B].insert(C,val);A.n_data+=1;A._rebuild_indexing()
	def delete(A,idx):
		B,C=A._find_block(idx);A.block_arr[B].erase(C);A.n_data-=1
		if A.block_arr[B].size==0:del A.block_arr[B]
		A._rebuild_indexing()
	def query_min_prefix(I,l,r):
		J=0;B=float(_I);A=0;C=0;B=float(_I);A=0
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
	def query_max_prefix(I,l,r):
		J=0;B=float(_Q);A=0;C=0;B=float(_Q);A=0
		for D in I.block_arr:
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
		if B<0 or B>=A.n_data:raise IndexError(_S)
		C,D=A._find_block(B);return A.block_arr[C].arr[D]
	def get_data_segment(C,l,r):
		if l<0 or r<0 or l>r or r>C.n_data:raise IndexError('Invalid segment range')
		D=[];A=0
		for E in C.block_arr:
			B=E.size
			if A>=r:break
			if A+B<=l:A+=B;continue
			F=max(0,l-A);G=min(B,r-A)
			if G>F:D.extend(E.arr[F:G])
			A+=B
		return D
	def get_data(A):return A.get_data_segment(0,A.n_data)
def cost_decrement_intra_swap(partial,route_idx,a_idx,b_idx):
	E=partial;C=b_idx;B=a_idx;assert B!=C,'Indices to swap must be different.'
	if B>C:B,C=C,B
	A=E.routes[route_idx];assert A[B]!=0 and A[C]!=0,_T;D=E.problem.D
	if B<C-1:F=D[A[B-1]][A[B]]+D[A[B]][A[B+1]]+D[A[C-1]][A[C]]+D[A[C]][A[C+1]]-D[A[B-1]][A[C]]-D[A[C]][A[B+1]]-D[A[C-1]][A[B]]-D[A[B]][A[C+1]]
	else:F=D[A[B-1]][A[B]]+D[A[B]][A[C]]+D[A[C]][A[C+1]]-D[A[B-1]][A[C]]-D[A[C]][A[B]]-D[A[B]][A[C+1]]
	return F
def intra_swap_one_route_operator(partial,route_idx,steps=_B,mode=_J,uplift=1,seed=42,verbose=_A):
	O=mode;K=steps;G=route_idx;a=random.Random(seed);B=partial.copy();A=B.problem;X=A.K;C=B.routes[G];F=len(C)
	if F<3:return B,[_A]*X,0
	if K is _B:K=F**2
	H={B:A for(A,B)in enumerate(C)};P=[0]*F;D=[0]*F;Q=[0]*F;E=[0]*F;Y=0;Z=0
	for J in range(F):
		I=C[J];L=0;M=0
		if A.is_ppick(I):L=1
		elif A.is_pdrop(I):L=-1
		elif A.is_parc_pick(I):R=A.rev_parc_pick(I);M=A.q[R-1]
		elif A.is_parc_drop(I):R=A.rev_parc_drop(I);M=-A.q[R-1]
		Y+=L;Z+=M;P[J]=Y;D[J]=L;Q[J]=Z;E[J]=M
	S=TreeSegment(data=P,op=min,identity=float(_I),sum_like=_A);T=TreeSegment(data=P,op=max,identity=0,sum_like=_A);U=TreeSegment(data=Q,op=min,identity=float(_I),sum_like=_A);V=TreeSegment(data=Q,op=max,identity=0,sum_like=_A)
	def c(a,b):
		assert a!=b
		def B(idx):
			A=idx
			if A==a:return b
			if A==b:return a
			return A
		def D(idx_old):
			F=idx_old;D=C[F]
			if A.is_ppick(D):
				J=A.rev_ppick(D);E=A.pdrop(J);G=H.get(E)
				if G is _B:return _C
				return B(F)<B(G)
			if A.is_pdrop(D):
				J=A.rev_pdrop(D);E=A.ppick(J);I=H.get(E)
				if I is _B:return _A
				return B(I)<B(F)
			if A.is_parc_pick(D):
				K=A.rev_parc_pick(D);E=A.parc_drop(K);G=H.get(E)
				if G is _B:return _C
				return B(F)<B(G)
			if A.is_parc_drop(D):
				K=A.rev_parc_drop(D);E=A.parc_pick(K);I=H.get(E)
				if I is _B:return _A
				return B(I)<B(F)
			return _A
		return D(a)and D(b)
	def d(a,b):
		assert a!=b
		if a>b:a,b=b,a
		A=D[b]-D[a]
		if A>0:return T.query(a,b)+A<=1
		elif A<0:return S.query(a,b)+A>=0
		else:return _C
	def e(a,b):
		assert a!=b
		if a>b:a,b=b,a
		B=E[b]-E[a]
		if B>0:return V.query(a,b)<=A.Q[G]-B
		elif B<0:return U.query(a,b)>=-B
		else:return _C
	def b(a,b):
		if not(c(a,b)and d(a,b)and e(a,b)):return _A,0
		A=cost_decrement_intra_swap(B,G,a,b);return _C,A
	def f():
		for A in range(1,F-1):
			for B in range(A+1,F-1):
				D,C=b(A,B)
				if not D or C<uplift:continue
				if O==_J:yield(A,B,C);return
				else:yield(A,B,C)
	def g():
		A=list(f())
		if not A:return
		if O==_U:return a.choice(A)
		elif O==_R:return max(A,key=lambda x:x[2])
		else:return A[0]
	N=0;h=0;W=[_A]*X
	def i(action):nonlocal C;nonlocal h;nonlocal W;nonlocal B;A,D,E=action;C[A],C[D]=C[D],C[A];B.route_costs[G]-=E;B.max_cost=max(B.max_cost,B.route_costs[G])
	def j(action):
		nonlocal C;nonlocal H;nonlocal D;nonlocal E;nonlocal S;nonlocal U;nonlocal T;nonlocal V;A,B,I=action
		if A>B:A,B=B,A
		F=D[B]-D[A]
		if F!=0:S.update(A,B,F);T.update(A,B,F)
		G=E[B]-E[A]
		if G!=0:U.update(A,B,G);V.update(A,B,G)
		D[A],D[B]=D[B],D[A];E[A],E[B]=E[B],E[A];H[C[A]]=A;H[C[B]]=B
	def k():
		nonlocal N
		while _C:
			if K is not _B and N>=K:break
			A=g()
			if A is _B:break
			i(A);j(A);N+=1;W[G]=_C
			if verbose:B,D,E=A;print(f"[Route {G}] Swapped positions {B} and {D} "+f"(nodes {C[D]} and {C[B]}). Cost decrease: {E}.")
	k();return B,W,N
def intra_swap_operator(partial,steps=_B,mode=_J,uplift=1,seed=42,verbose=_A):
	F=verbose;C=steps;B=partial
	if C is _B:C=10**9
	D=0;G=[_A]*B.problem.K;E=B.copy()
	for A in range(B.problem.K):
		I,J,H=intra_swap_one_route_operator(E,route_idx=A,steps=C-D,mode=mode,uplift=uplift,seed=seed,verbose=F);E=I;D+=H
		if J[A]:G[A]=_C
		if F:print(f"Route {A}: performed {H} intra-route swaps.")
	return E,G,D
def cost_decrement_inter_swap(partial,route_a_idx,route_b_idx,p_idx_a,d_idx_a,p_idx_b,d_idx_b):
	J=route_b_idx;I=route_a_idx;H=partial;G=d_idx_b;F=d_idx_a;E=p_idx_b;D=p_idx_a;A=H.routes[I];B=H.routes[J];assert A[D]!=0 and B[E]!=0,_T;Q=H.route_costs[I];R=H.route_costs[J];S=H.max_cost;C=H.problem.D
	if D+1==F:K=C[A[D-1]][A[D]]+C[A[D]][A[F]]+C[A[F]][A[F+1]];L=C[A[D-1]][B[E]]+C[B[E]][B[G]]+C[B[G]][A[F+1]]
	else:K=C[A[D-1]][A[D]]+C[A[D]][A[D+1]]+C[A[F-1]][A[F]]+C[A[F]][A[F+1]];L=C[A[D-1]][B[E]]+C[B[E]][A[D+1]]+C[A[F-1]][B[G]]+C[B[G]][A[F+1]]
	if E+1==G:M=C[B[E-1]][B[E]]+C[B[E]][B[G]]+C[B[G]][B[G+1]];N=C[B[E-1]][A[D]]+C[A[D]][A[F]]+C[A[F]][B[G+1]]
	else:M=C[B[E-1]][B[E]]+C[B[E]][B[E+1]]+C[B[G-1]][B[G]]+C[B[G]][B[G+1]];N=C[B[E-1]][A[D]]+C[A[D]][B[E+1]]+C[B[G-1]][A[F]]+C[A[F]][B[G+1]]
	O=Q-K+L;P=R-M+N;T=max(O,P,*(H.route_costs[A]for A in range(H.problem.K)if A!=I and A!=J));return O,P,S-T
def inter_swap_route_pair_operator(partial,route_a_idx,route_b_idx,steps=_B,mode=_J,uplift=1,seed=42,verbose=_A):
	P=mode;O=steps;F=route_b_idx;E=route_a_idx;c=random.Random(seed);B=partial.copy();A=B.problem;C=B.routes[E];D=B.routes[F];Q=len(C);R=len(D)
	if Q<3 or R<3:return B,[_A]*A.K,0
	def a(route):
		G=route;C=len(G);H=[0]*C;K=[0]*C;I=[0]*C;L=[0]*C;M=0;N=0
		for(D,B)in enumerate(G):
			E=0;F=0
			if A.is_ppick(B):E=1
			elif A.is_pdrop(B):E=-1
			elif A.is_parc_pick(B):J=A.rev_parc_pick(B);F=A.q[J-1]
			elif A.is_parc_drop(B):J=A.rev_parc_drop(B);F=-A.q[J-1]
			M+=E;N+=F;H[D]=M;K[D]=E;I[D]=N;L[D]=F
		O=TreeSegment(data=H,op=min,identity=float(_I),sum_like=_A);P=TreeSegment(data=H,op=max,identity=0,sum_like=_A);Q=TreeSegment(data=I,op=min,identity=float(_I),sum_like=_A);R=TreeSegment(data=I,op=max,identity=0,sum_like=_A);S={B:A for(A,B)in enumerate(G)};return S,K,L,(O,P,Q,R)
	K,G,H,d=a(C);L,I,J,e=a(D);S,T,U,V=d;W,X,Y,Z=e;f=A.Q[E];g=A.Q[F]
	def h(p_idx_a,q_idx_a,p_idx_b,q_idx_b):
		F=q_idx_b;E=q_idx_a;B=p_idx_b;A=p_idx_a;C=I[B]-G[A]
		if C!=0:
			H=S.query(A,E);J=T.query(A,E)
			if H+C<0 or J+C>1:return _A
		D=G[A]-I[B]
		if D!=0:
			K=W.query(B,F);L=X.query(B,F)
			if K+D<0 or L+D>1:return _A
		return _C
	def i(p_idx_a,q_idx_a,p_idx_b,q_idx_b):
		F=q_idx_b;E=q_idx_a;B=p_idx_b;A=p_idx_a;C=J[B]-H[A]
		if C!=0:
			G=U.query(A,E);I=V.query(A,E)
			if G+C<0 or I+C>f:return _A
		D=H[A]-J[B]
		if D!=0:
			K=Y.query(B,F);L=Z.query(B,F)
			if K+D<0 or L+D>g:return _A
		return _C
	def j(p_idx_a,q_idx_a,p_idx_b,q_idx_b):
		G=q_idx_b;D=p_idx_b;C=q_idx_a;A=p_idx_a
		if not h(A,C,D,G):return _A,0,0,0
		if not i(A,C,D,G):return _A,0,0,0
		H,I,J=cost_decrement_inter_swap(B,E,F,A,C,D,G);return _C,H,I,J
	def k():
		T=[B for B in range(1,Q-1)if A.is_ppick(C[B])or A.is_parc_pick(C[B])];U=[B for B in range(1,R-1)if A.is_ppick(D[B])or A.is_parc_pick(D[B])]
		for B in T:
			H=C[B]
			if A.is_ppick(H):V=A.rev_ppick(H);M=A.pdrop(V)
			else:W=A.rev_parc_pick(H);M=A.parc_drop(W)
			E=K.get(M)
			if E is _B:continue
			for F in U:
				I=D[F]
				if A.is_ppick(I):X=A.rev_ppick(I);N=A.pdrop(X)
				else:Y=A.rev_parc_pick(I);N=A.parc_drop(Y)
				G=L.get(N)
				if G is _B:continue
				Z,O,S,J=j(B,E,F,G)
				if not Z or J<uplift:continue
				if P==_J:yield(B,E,F,G,O,S,J);return
				else:yield(B,E,F,G,O,S,J)
	def l():
		A=list(k())
		if not A:return
		if P==_U:return c.choice(A)
		elif P==_R:return max(A,key=lambda x:x[4])
		else:return A[0]
	M=0;b=0;N=[_A]*A.K
	if O is _B:O=(Q+R)**2
	def m(action):nonlocal C,D,B;A,G,H,I,J,K,L=action;M,N=C[A],C[G];O,P=D[H],D[I];C[A],C[G]=O,P;D[H],D[I]=M,N;B.route_costs[E]=J;B.route_costs[F]=K;B.max_cost-=L
	def n(action):
		nonlocal K,L;nonlocal G,I;nonlocal H,J;nonlocal S,T,U,V;nonlocal W,X,Y,Z;A,E,B,F,Q,__,___=action;M=I[B]-G[A];N=J[B]-H[A]
		if M!=0:S.update(A,E,M);T.update(A,E,M)
		if N!=0:U.update(A,E,N);V.update(A,E,N)
		O=G[A]-I[B];P=H[A]-J[B]
		if O!=0:W.update(B,F,O);X.update(B,F,O)
		if P!=0:Y.update(B,F,P);Z.update(B,F,P)
		G[A],I[B]=I[B],G[A];G[E],I[F]=I[F],G[E];H[A],J[B]=J[B],H[A];H[E],J[F]=J[F],H[E];K[C[A]]=A;K[C[E]]=E;L[D[B]]=B;L[D[F]]=F
	def o():
		nonlocal M,N,b
		while M<O:
			A=l()
			if A is _B:break
			m(A);n(A);b+=A[6];N[E]=_C;N[F]=_C;M+=1
			if verbose:B,C,D,G,I,__,H=A;print(f"[Routes {E} & {F}] "+f"Swapped nodes at positions ({B}, {C}) and ({D}, {G}). "+f"Cost decrease: {H}.")
	o();return B,N,M
def inter_swap_operator(partial,steps=_B,mode=_J,uplift=1,seed=42,verbose=_A):
	N=verbose;H=steps;G=partial;T=random.Random(seed);I=G.problem.K
	if I<2:return G.copy(),[_A]*I,0
	A=G.copy();J=[_A]*I;E=0;O=H if H is not _B else 10**9;F=[(-B,A)for(A,B)in enumerate(A.route_costs)];C=[(B,A)for(A,B)in enumerate(A.route_costs)];heapq.heapify(F);heapq.heapify(C)
	def U():
		while F:
			B,C=heapq.heappop(F)
			if-B==A.route_costs[C]:return-B,C
	def V(exclude_idx=_B):
		while C:
			D,B=heapq.heappop(C)
			if B==exclude_idx:continue
			if D==A.route_costs[B]:return D,B
	def K(idx):B=idx;D=A.route_costs[B];heapq.heappush(F,(-D,B));heapq.heappush(C,(D,B))
	while _C:
		if H is not _B and E>=O:break
		P=U()
		if P is _B:break
		W,B=P;Q=[];R=_A
		while _C:
			L=V(exclude_idx=B)
			if L is _B:break
			W,D=L;Q.append(L);X,S,M=inter_swap_route_pair_operator(A,route_a_idx=B,route_b_idx=D,steps=O-E,mode=mode,uplift=uplift,seed=T.randint(10,10**9),verbose=N)
			if M>0:
				A=X;K(B);K(D);E+=M
				if S[B]:J[B]=_C
				if S[D]:J[D]=_C
				R=_C
				if N:print(f"Inter-route swap between routes {B} and {D} "+f"performed {M} swaps.")
				break
		for(Y,Z)in Q:heapq.heappush(C,(Y,Z))
		if not R:K(B);break
	return A,J,E
def read_instance():
	A,B,D=map(int,sys.stdin.readline().strip().split());E=list(map(int,sys.stdin.readline().split()));F=list(map(int,sys.stdin.readline().split()));C=[[0]*(2*A+2*B+1)for C in range(2*A+2*B+1)]
	for G in range(2*A+2*B+1):H=sys.stdin.readline().strip();C[G]=list(map(int,H.split()))
	return ShareARideProblem(A,B,D,E,F,C)
def main(verbose=_A):
	B=verbose;C=read_instance();A,J=iterative_greedy_balanced_solver(prob=C,iterations=100000,time_limit=6e1,seed=42,verbose=_A,destroy_proba=.5,destroy_steps=min(6,C.num_nodes//(2*C.K)+1),destroy_T=_E,rebuild_proba=.25,rebuild_steps=2,rebuild_T=1e1);assert A is not _B,'No solution found by IG solver.'
	if B:print(f"Initial solution cost: {A.max_cost:.2f}");print();print()
	G=time.time();D=PartialSolution.from_solution(A);E,H,F=inter_swap_operator(partial=D,steps=_B,mode=_J,seed=100,verbose=_A);A=E.to_solution();assert A,'No solution found after int.'
	if B:print(f"Total inter-swap performed: {F}");print(f"Cost after inter-swap: {A.max_cost:.2f}");print(f"Time for inter-swap: {time.time()-G:.2f} seconds");print();print()
	I=time.time();D=PartialSolution.from_solution(A);E,H,F=intra_swap_operator(partial=D,steps=_B,mode=_R,seed=200,verbose=_A);A=E.to_solution();assert A,'No solution found after relocates.'
	if B:print(f"Total relocates performed: {F}");print(f"Cost after relocates: {A.max_cost:.2f}");print(f"Time for relocates: {time.time()-I:.2f} seconds");print();print()
	A.stdin_print(verbose=_A)
if __name__=='__main__':main(verbose=_A)