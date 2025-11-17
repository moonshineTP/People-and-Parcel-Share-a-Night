_Z='MCTSNode'
_Y='Cannot swap depot nodes.'
_X='Index out of bounds'
_W='----------------'
_V='return'
_U='best'
_T='stochastic'
_S='-inf'
_R='actions_evaluated'
_Q='actions'
_P='time'
_O='load'
_N='route'
_M='iterations'
_L='inf'
_K=.0
_J='parcels'
_I='pos'
_H=1.
_G='first'
_F='cost'
_E='passenger'
_D='ended'
_C=True
_B=None
_A=False
import sys,time,random
from typing import List,Optional,Tuple,Dict,Any
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
		E=route_costs;C=routes;A=problem
		if not C:raise ValueError('Routes list cannot be empty.')
		if len(C)!=A.K:raise ValueError(f"Expected {A.K} routes, got {len(C)}.")
		if not E:D=[route_cost_from_sequence(B,A.D)for B in C]
		else:D=E
		B.problem=A;B.routes=C;B.route_costs=D;B.n_actions=2*(A.N+A.M)+A.K;B.max_cost=max(D)if D else 0
	def is_valid(H):
		A=H.problem;N,O,L=A.N,A.M,A.K
		if len(H.routes)!=L:return _A
		for(M,I)in enumerate(H.routes):
			if not(I[0]==0 and I[-1]==0):return _A
			E=set();G=set();F=0;J=set();K=set()
			for C in I[1:-1]:
				if A.is_ppick(C):
					D=A.rev_ppick(C)
					if D in J:return _A
					if len(E)>=1:return _A
					E.add(D);J.add(D)
				elif A.is_pdrop(C):
					D=A.rev_pdrop(C)
					if D not in E:return _A
					E.remove(D)
				elif A.is_parc_pick(C):
					B=A.rev_parc_pick(C)
					if B in K or B in G:return _A
					F+=A.q[B-1]
					if F>A.Q[M]:return _A
					K.add(B);G.add(B)
				elif A.is_parc_drop(C):
					B=A.rev_parc_drop(C)
					if B not in G:return _A
					assert F-A.q[B-1]>=0;F-=A.q[B-1];G.remove(B)
			if E:return _A
			if F!=0:return _A
		return _C
	def stdin_print(A,verbose=_A):
		B=verbose;assert len(A.routes)==len(A.route_costs);print(A.problem.K)
		for(C,D)in zip(A.routes,A.route_costs):
			print(len(C));print(' '.join(map(str,C)))
			if B:print(f"// Route cost: {D}");print(_W)
		if B:print(f"//// Max route cost: {A.max_cost} ////")
class PartialSolution:
	def __init__(A,problem,routes):C=routes;B=problem;A.problem=B;A.routes=A._init_routes(C);A.route_costs=A._init_costs(C);A.max_cost=max(A.route_costs);A.avg_cost=sum(A.route_costs)/B.K;A.node_assignment=A._init_node_assignment();A.remaining_pass_pick,A.remaining_pass_drop,A.remaining_parc_pick,A.remaining_parc_drop,A.route_states=A._init_states();A.n_actions=sum(len(A)-1 for A in A.routes)
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
		for(O,D)in enumerate(H.routes):
			F=0;G=set();K=0
			for C in D[1:]:
				if A.is_ppick(C):E=A.rev_ppick(C);L.discard(E);I.add(E);F=E
				elif A.is_pdrop(C):
					E=A.rev_pdrop(C);I.discard(E)
					if F==E:F=0
				elif A.is_parc_pick(C):B=A.rev_parc_pick(C);M.discard(B);J.add(B);G.add(B);K+=A.q[B-1]
				elif A.is_parc_drop(C):
					B=A.rev_parc_drop(C)
					if B in G:G.remove(B);K-=A.q[B-1]
					J.discard(B)
			P=D[-1];Q=len(D)>1 and D[-1]==0;R={_N:D,_I:P,_F:H.route_costs[O],_O:K,_E:F,_J:G.copy(),_Q:len(D)-1,_D:Q};N.append(R)
		return L,I,M,J,N
	def is_valid(B):
		A=B.problem;W,X,Q=A.N,A.M,A.K
		if not len(B.routes)==len(B.route_states)==len(B.route_costs)==Q:return _A
		if len(B.node_assignment)!=len(A.D):return _A
		R=set(range(1,W+1));M=set();S=set(range(1,X+1));N=set();O=[-1]*len(A.D);T=0;P=0;Y=0
		for I in range(Q):
			D=B.routes[I];F=B.route_states[I]
			if not D or D[0]!=0:return _A
			if F[_N]!=D:return _A
			if F[_I]!=D[-1]:return _A
			if F[_Q]!=len(D)-1:return _A
			Z=len(D)>1 and D[-1]==0
			if F[_D]!=Z:return _A
			H=set();J=set();L=0;U=D[0];K=0
			for C in D[1:]:
				if not 0<=C<len(A.D):return _A
				K+=A.D[U][C];U=C
				if C!=0:
					V=O[C]
					if V!=-1 and V!=I:return _A
					O[C]=I
				if A.is_ppick(C):
					G=A.rev_ppick(C)
					if G in H or H:return _A
					H.add(G);R.discard(G);M.add(G)
				elif A.is_pdrop(C):
					G=A.rev_pdrop(C)
					if G not in H:return _A
					H.remove(G);M.discard(G)
				elif A.is_parc_pick(C):
					E=A.rev_parc_pick(C)
					if E in J:return _A
					L+=A.q[E-1]
					if L>A.Q[I]:return _A
					J.add(E);S.discard(E);N.add(E)
				elif A.is_parc_drop(C):
					E=A.rev_parc_drop(C)
					if E not in J:return _A
					L-=A.q[E-1];J.remove(E);N.discard(E)
			a=next(iter(H))if H else 0
			if F[_E]!=a:return _A
			if F[_J]!=J:return _A
			if F[_O]!=L:return _A
			if F[_F]!=K or B.route_costs[I]!=K:return _A
			T+=len(D)-1;P=max(P,K);Y+=K
		if R!=B.remaining_pass_pick:return _A
		if M!=B.remaining_pass_drop:return _A
		if S!=B.remaining_parc_pick:return _A
		if N!=B.remaining_parc_drop:return _A
		if O!=B.node_assignment:return _A
		if B.max_cost!=P:return _A
		if B.n_actions!=T:return _A
		return _C
	def is_identical(A,other):
		B=other
		if A is B:return _C
		if A.problem is not B.problem:return _A
		if A.n_actions!=B.n_actions:return _A
		def C(ps):
			A=[];B=[];D={}
			for(C,E)in enumerate(ps.node_assignment):
				if C==0:continue
				if E==-1:B.append(C)
				else:D.setdefault(E,[]).append(C)
			for F in D.values():A.append(tuple(sorted(F)))
			A.sort();B.sort();return tuple(A),tuple(B)
		if C(A)!=C(B):return _A
		def D(ps):
			B=[]
			for(C,A)in enumerate(ps.routes):D=A[1]if len(A)>1 else-1;E=A[2]if len(A)>2 else-1;B.append((D,E,ps.route_costs[C]))
			B.sort();return B
		if D(A)!=D(B):return _A
		return _C
	def copy(A):return PartialSolution(problem=A.problem,routes=[A.copy()for A in A.routes])
	def stdin_print(A,verbose=_A):
		B=verbose;assert len(A.routes)==len(A.route_costs);print(A.problem.K)
		for(C,D)in zip(A.routes,A.route_costs):
			print(len(C));print(' '.join(map(str,C)))
			if B:print(f"// Route cost: {D}");print(_W)
		if B:print(f"//// Max route cost: {A.max_cost} ////")
	def possible_actions(G,t_idx):
		I=t_idx;C=G.route_states[I]
		if C[_D]:return[]
		A=G.problem;H=C[_I];D=[]
		if C[_E]==0:
			for F in list(G.remaining_pass_pick):B=A.D[H][A.ppick(F)];D.append(('pickP',F,B))
		else:F=C[_E];B=A.D[H][A.pdrop(F)];D.append(('dropP',F,B))
		for E in list(G.remaining_parc_pick):
			J=A.q[E-1]
			if C[_O]+J<=A.Q[I]:B=A.D[H][A.parc_pick(E)];D.append(('pickL',E,B))
		for E in list(C[_J]):B=A.D[H][A.parc_drop(E)];D.append(('dropL',E,B))
		D.sort(key=lambda x:x[2]);return D
	def apply_action(C,t_idx,kind,node_idx,inc):
		G=kind;D=t_idx;A=node_idx;B=C.route_states[D]
		if B[_D]:raise ValueError(f"Cannot apply action on ended route {D}.")
		E=C.problem
		if G=='pickP':
			if B[_E]!=0:raise ValueError(f"Taxi {D} already has passenger {B[_E]}.")
			F=E.ppick(A);B[_E]=A;C.remaining_pass_pick.discard(A);C.remaining_pass_drop.add(A)
		elif G=='dropP':
			if B[_E]!=A:raise ValueError(f"Taxi {D} is not carrying passenger {A}.")
			F=E.pdrop(A);B[_E]=0;C.remaining_pass_drop.discard(A)
		elif G=='pickL':
			H=E.q[A-1]
			if B[_O]+H>E.Q[D]:raise ValueError(f"Taxi {D} capacity exceeded for parcel {A}.")
			F=E.parc_pick(A);B[_O]+=H;B[_J].add(A);C.remaining_parc_pick.discard(A);C.remaining_parc_drop.add(A)
		elif G=='dropL':
			if A not in B[_J]:raise ValueError(f"Taxi {D} does not carry parcel {A}.")
			F=E.parc_drop(A);B[_O]-=E.q[A-1];B[_J].discard(A);C.remaining_parc_drop.discard(A)
		else:raise ValueError(f"Unknown action kind: {G}")
		B[_N].append(F);B[_F]+=inc;B[_I]=F;B[_Q]+=1;C.node_assignment[F]=D;C.route_costs[D]=B[_F];C.max_cost=max(C.max_cost,B[_F]);C.avg_cost=sum(C.route_costs)/C.problem.K;C.n_actions+=1
	def apply_return_to_depot(B,t_idx):
		C=t_idx;A=B.route_states[C]
		if A[_D]:return
		if A[_I]==0 and len(A[_N])>1:A[_D]=_C;return
		if A[_E]!=0 or A[_J]:raise ValueError(f"Taxi {C} must drop all loads before returning to depot.")
		A[_F]+=B.problem.D[A[_I]][0];A[_N].append(0);A[_I]=0;A[_Q]+=1;A[_D]=_C;B.route_costs[C]=A[_F];B.max_cost=max(B.max_cost,A[_F]);B.avg_cost=sum(B.route_costs)/B.problem.K;B.n_actions+=1
	def reverse_action(A,t_idx):
		G=t_idx;B=A.route_states[G]
		if len(B[_N])<=1:raise ValueError(f"No actions to reverse for taxi {G}.")
		C=B[_N].pop();H=B[_N][-1];I=A.problem.D[H][C];B[_F]-=I;B[_I]=H;B[_Q]-=1;B[_D]=_A;D=A.problem
		if D.is_ppick(C):F=D.rev_ppick(C);B[_E]=0;A.remaining_pass_pick.add(F);A.remaining_pass_drop.discard(F)
		elif D.is_pdrop(C):F=D.rev_pdrop(C);B[_E]=F;A.remaining_pass_pick.discard(F);A.remaining_pass_drop.add(F)
		elif D.is_parc_pick(C):E=D.rev_parc_pick(C);B[_O]-=D.q[E-1];B[_J].discard(E);A.remaining_parc_pick.add(E);A.remaining_parc_drop.discard(E)
		elif D.is_parc_drop(C):E=D.rev_parc_drop(C);B[_O]+=D.q[E-1];B[_J].add(E);A.remaining_parc_pick.discard(E);A.remaining_parc_drop.add(E)
		else:B[_D]=_A
		A.route_costs[G]=B[_F];A.max_cost=max(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.node_assignment[C]=-1;A.n_actions-=1
	def is_complete(A):return all(A[_D]for A in A.route_states)
	def to_solution(A):
		if not A.is_complete():print('Cannot convert to Solution: not all routes have ended at depot.');return
		B=Solution(problem=A.problem,routes=A.routes,route_costs=A.route_costs)
		if B is _B or not B.is_valid():print('Warning: Converted solution is not valid.')
		return B
	@staticmethod
	def from_solution(sol):A=[A.copy()for A in sol.routes];return PartialSolution(problem=sol.problem,routes=A)
class PartialSolutionSwarm:
	def __init__(A,solutions=_B,n_partials=_B):
		C=n_partials;B=solutions
		if not B:
			if C is _B or C<=0:raise ValueError('Must provide either solutions list or positive n_partials.')
			A.parsol_list=[];A.parsol_nact=[];A.costs=[];A.min_cost=0;A.max_cost=0;A.avg_cost=_K;A.best_parsol=_B;return
		A.parsol_list=B;A.parsol_nact=[A.n_actions for A in B];A.costs=[A.max_cost for A in B];A.min_cost=min(A.costs);A.max_cost=max(A.costs);A.avg_cost=sum(A.max_cost for A in B)/len(B);A.best_parsol=min(B,key=lambda s:s.max_cost)
	def apply_action_one(A,sol_idx,t_idx,kind,node_idx,inc):
		C=sol_idx;B=A.parsol_list[C];B.apply_action(t_idx,kind,node_idx,inc);A.parsol_nact[C]=B.n_actions;A.costs[C]=B.max_cost;A.min_cost=min(A.costs);A.max_cost=max(A.costs);A.avg_cost=sum(A.costs)/len(A.costs)
		if B.max_cost==A.min_cost:A.best_parsol=B
	def apply_return_to_depot_one(A,sol_idx,t_idx):
		C=sol_idx;B=A.parsol_list[C];B.apply_return_to_depot(t_idx);A.parsol_nact[C]=B.n_actions;A.costs[C]=B.max_cost;A.min_cost=min(A.costs);A.max_cost=max(A.costs);A.avg_cost=sum(A.costs)/len(A.costs)
		if B.max_cost==A.min_cost:A.best_parsol=B
	def copy(A):B=[A.copy()for A in A.parsol_list];return PartialSolutionSwarm(solutions=B)
	def extract_best_solution(A):
		if A.best_parsol and A.best_parsol.is_complete():return A.best_parsol.to_solution()
from typing import Iterator,List,Tuple,Optional,Callable,Union,Sequence
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
			A.size=len(A.arr);A.sum=sum(A.arr);B=0;C=float(_L);D=float(_S)
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
		B=idx;assert A.block_arr,'No blocks present';assert 0<=B<A.n_data,_X
		if B>A.n_data:B=A.n_data
		C=bisect.bisect_right(A.block_prefix,B)-1;D=B-A.block_prefix[C];return C,D
	def insert(A,idx,val):
		B=val
		if idx==A.n_data:
			if not A.block_arr:A.block_arr.append(A.Block([B]))
			else:
				C=A.block_arr[-1]
				if C.size>=2*A.block_size:A.block_arr.append(A.Block([B]))
				else:C.insert(C.size,B)
			A.n_data+=1;A._rebuild_indexing();return
		D,H=A._find_block(idx);E=A.block_arr[D];E.insert(H,B)
		if E.size>2*A.block_size:F=E.arr;G=len(F)//2;I=A.Block(F[:G]);J=A.Block(F[G:]);A.block_arr[D:D+1]=[I,J]
		A.n_data+=1;A._rebuild_indexing()
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
		A.n_data-=1;A._rebuild_indexing()
	def query_min_prefix(I,l,r):
		B=float(_L);A=0;C=0;B=float(_L);A=0
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
		B=float(_S);A=0;C=0;B=float(_S);A=0
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
		if B<0 or B>=A.n_data:raise IndexError(_X)
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
def cost_decrement_relocate(partial,from_route_idx,to_route_idx,p_idx_from,q_idx_from,p_idx_to,q_idx_to):
	P=to_route_idx;O=from_route_idx;K=q_idx_to;J=p_idx_to;H=q_idx_from;G=p_idx_from;D=partial;C=D.routes[O];I=D.routes[P];assert O!=P,'from_route_idx and to_route_idx must be different for relocate.';assert C[G]!=0 and C[H]!=0,'Cannot relocate depot nodes.';assert 1<=G<H,'Invalid pickup/drop indices in from_route.';assert 1<=J<K,'Invalid pickup/drop indices in to_route';A=D.problem.D;Y=D.max_cost;E=C[G];F=C[H]
	if G+1==H:B=C[G-1];L=C[H+1];Q=A[B][E]+A[E][F]+A[F][L];R=A[B][L]
	else:B=C[G-1];M=C[G+1];S=C[H-1];L=C[H+1];Q=A[B][E]+A[E][M]+A[S][F]+A[F][L];R=A[B][M]+A[S][L]
	T=D.route_costs[O]-Q+R
	if K==J+1:B=I[J-1];N=I[K-1];U=A[B][N];V=A[B][E]+A[E][F]+A[F][N]
	else:B=I[J-1];M=I[J];W=I[K-2];N=I[K-1];U=A[B][M]+A[W][N];V=A[B][E]+A[E][M]+A[W][F]+A[F][N]
	X=D.route_costs[P]+V-U;Z=max(T,X,*(D.route_costs[A]for A in range(D.problem.K)if A!=O and A!=P));return T,X,Y-Z
def relocate_from_to(partial,from_route_idx,to_route_idx,steps,mode,uplift=1,seed=42,verbose=_A):
	N=partial;M=mode;F=to_route_idx;E=from_route_idx;P=random.Random(seed);A=N.problem;C=N.copy();B=C.routes[E];D=C.routes[F];G=len(B);H=len(D)
	if G<5:return N,[_A]*A.K,0
	def O(route,n):
		F=[0]*n;G=[0]*n
		for(H,B)in enumerate(route):
			C=0;D=0
			if A.is_ppick(B):C=1
			elif A.is_pdrop(B):C=-1
			elif A.is_parc_pick(B):E=A.rev_parc_pick(B);D=A.q[E-1]
			elif A.is_parc_drop(B):E=A.rev_parc_drop(B);D=-A.q[E-1]
			F[H]=C;G[H]=D
		I=MinMaxPfsumArray(F);J=MinMaxPfsumArray(G);return I,J
	I,J=O(B,G);K,L=O(D,H);Q=A.Q[E];R=A.Q[F]
	def S(p_idx_a,q_idx_a,p_idx_b,q_idx_b):
		G=q_idx_b;F=p_idx_b;E=q_idx_a;D=p_idx_a;H=B[D];C=1 if A.is_ppick(H)else 0
		if C==0:return _C
		J=I.query_min_prefix(D,E);L=I.query_max_prefix(D,E)
		if J-C<0 or L-C>1:return _A
		M=K.query_min_prefix(F-1,G-1);N=K.query_max_prefix(F-1,G-1)
		if M+C<0 or N+C>1:return _A
		return _C
	def T(p_idx_a,q_idx_a,p_idx_b,q_idx_b):
		G=q_idx_b;F=p_idx_b;E=q_idx_a;D=p_idx_a;H=B[D]
		if A.is_parc_pick(H):I=A.rev_parc_pick(H);C=A.q[I-1]
		else:C=0
		if C==0:return _C
		K=J.query_min_prefix(D,E);M=J.query_max_prefix(D,E)
		if K-C<0 or M-C>Q:return _A
		N=L.query_min_prefix(F-1,G-1);O=L.query_max_prefix(F-1,G-1)
		if N+C<0 or O+C>R:return _A
		return _C
	def U(p_idx_a,q_idx_a,p_idx_b,q_idx_b):
		G=q_idx_b;D=p_idx_b;B=q_idx_a;A=p_idx_a
		if not S(A,B,D,G):return _A,0,0,0
		if not T(A,B,D,G):return _A,0,0,0
		H,I,J=cost_decrement_relocate(C,E,F,A,B,D,G);return _C,H,I,J
	def V():
		O={B:A for(A,B)in enumerate(B)};P=[C for C in range(1,G-1)if A.is_ppick(B[C])or A.is_parc_pick(B[C])];Q=[(A,B)for A in range(1,H)for B in range(A+1,H+1)]
		for C in P:
			E=B[C]
			if A.is_ppick(E):R=A.rev_ppick(E);K=A.pdrop(R)
			else:S=A.rev_parc_pick(E);K=A.parc_drop(S)
			D=O.get(K)
			if D is _B or D<=C:continue
			for(F,I)in Q:
				T,L,N,J=U(C,D,F,I)
				if not T or J<uplift:continue
				if M==_G:yield(C,D,F,I,L,N,J);return
				else:yield(C,D,F,I,L,N,J)
	def W():
		A=list(V())
		if not A:return
		if M==_T:return P.choice(A)
		elif M==_U:return max(A,key=lambda x:x[6])
		else:return A[0]
	def X(action):A,G,H,I,J,K,L=action;nonlocal B,D,C;M=B[A];N=B[G];del B[G];del B[A];D.insert(H,M);D.insert(I,N);C.routes[E]=B;C.routes[F]=D;C.route_costs[E]=J;C.route_costs[F]=K;C.max_cost-=L
	def Y(action):
		nonlocal I,J,K,L;nonlocal B,D;C,E,G,H,*M=action
		def F(node):
			B=node
			if A.is_ppick(B):return 1,0
			if A.is_pdrop(B):return-1,0
			if A.is_parc_pick(B):C=A.rev_parc_pick(B);return 0,A.q[C-1]
			if A.is_parc_drop(B):C=A.rev_parc_drop(B);return 0,-A.q[C-1]
			return 0,0
		I.delete(E);J.delete(E);I.delete(C);J.delete(C);K.insert(G,F(B[C])[0]);L.insert(G,F(B[C])[1]);K.insert(H,F(B[E])[0]);L.insert(H,F(B[E])[1])
	def Z():
		nonlocal G,H,B,D;I=0;J=[_A]*A.K
		while I<steps:
			C=W()
			if C is _B:break
			Y(C);X(C);I+=1;J[E]=_C;J[F]=_C;G-=2;H+=2
			if G<5:break
			if verbose:K,L,N,O,Q,__,P=C;print(f"[Relocate {E}->{F}] moved request (P:{K},D:{L}) to ({N},{O}) dec={P}")
			if M==_G:break
		return J,I
	a,b=Z();return C,a,b
def relocate_operator(partial,steps=_B,mode=_G,uplift=1,seed=42,verbose=_A):
	I=verbose;H=steps;F=partial;B=F.problem.K
	if B<2:return F.copy(),[_A]*B,0
	J=H if H is not _B else 10**9;O=random.Random(seed);A=F.copy();K=[_A]*B;C=0
	while C<J:
		L=list(enumerate(A.route_costs));D=max(L,key=lambda x:x[1])[0];P=[A for(A,B)in sorted(L,key=lambda x:x[1])][:max(4,B//2)]
		if len(A.routes[D])<5:break
		M=_A
		for E in P:
			if E==D:continue
			if len(A.routes[E])<2:continue
			Q=J-C;R,S,G=relocate_from_to(A,from_route_idx=D,to_route_idx=E,steps=Q,mode=mode,uplift=uplift,seed=O.randint(10,10**9),verbose=I)
			if G>0:
				A=R;C+=G
				for N in range(B):
					if S[N]:K[N]=_C
				M=_C
				if I:print(f"{G} relocation made from route {D} to route {E}")
				break
		if not M:break
	return A,K,C
def sample_from_weight(rng,weights):
	A=weights;C=sum(A)
	if C<1e-10:B=rng.randrange(len(A))
	else:
		E=rng.random()*C;D=_K;B=0
		for(F,G)in enumerate(A):
			D+=G
			if E<=D:B=F;break
	return B
from typing import List,Union
def softmax_weighter(incs,T):
	A=incs;B,E=min(A),max(A);C=E-B
	if C<1e-06:return[_H]*len(A)
	D=[]
	for F in A:G=(F-B)/C;D.append((_H-G+.1)**(_H/T))
	return D
def greedy_balanced_solver(problem,premature_routes=[],verbose=_A):
	G=verbose;M=time.time();A=PartialSolution(problem=problem,routes=premature_routes);E=A.route_states
	def N():return bool(A.remaining_pass_pick or A.remaining_pass_drop or A.remaining_parc_pick or A.remaining_parc_drop)
	C={_M:0,_R:0}
	while N():
		C[_M]+=1;H=[A for(A,B)in enumerate(E)if not B[_D]]
		if not H:break
		D=min(H,key=lambda i:E[i][_F]);F=A.possible_actions(D);C[_R]+=len(F)
		if not F:A.apply_return_to_depot(D);continue
		I,J,K=min(F,key=lambda x:x[2]);A.apply_action(D,I,J,K)
		if G:print(f"[Greedy] Taxi {D} extended route with {I} {J} (inc {K})")
	for(O,P)in enumerate(E):
		if not P[_D]:A.apply_return_to_depot(O)
	B=A.to_solution();L=time.time()-M;Q={_M:C[_M],_R:C[_R],_P:L}
	if B and not B.is_valid():B=_B
	assert B.is_valid()if B else _C
	if G:print('[Greedy] All tasks completed.');print(f"[Greedy] Solution max cost: {B.max_cost if B else'N/A'}");print(f"[Greedy] Time taken: {L:.4f} seconds")
	return B,Q
from typing import Any,List,Optional,Tuple,Dict
import heapq
from typing import List,Optional,Tuple
def cost_decrement_intra_swap(partial,route_idx,a_idx,b_idx):
	H=route_idx;G=partial;C=b_idx;B=a_idx;assert B!=C,'Indices to swap must be different.'
	if B>C:B,C=C,B
	A=G.routes[H];assert A[B]!=0 and A[C]!=0,_Y;D=G.problem.D
	def E(idx):return A[idx]if 0<=idx<=G.route_states[H][_Q]else _B
	def F(from_node,to_node):
		A=to_node
		if A is _B:return 0
		return D[from_node][A]
	if B<C-1:I=D[A[B-1]][A[B]]+F(A[B],E(B+1))+D[A[C-1]][A[C]]+F(A[C],E(C+1))-D[A[B-1]][A[C]]-F(A[C],E(B+1))-D[A[C-1]][A[B]]-F(A[B],E(C+1))
	else:I=D[A[B-1]][A[B]]+D[A[B]][A[C]]+F(A[C],E(C+1))-D[A[B-1]][A[C]]-D[A[C]][A[B]]-F(A[B],E(C+1))
	return I
def intra_swap_one_route_operator(partial,route_idx,steps=_B,mode=_G,uplift=1,seed=42,verbose=_A):
	O=mode;K=steps;G=route_idx;a=random.Random(seed);B=partial.copy();A=B.problem;X=A.K;C=B.routes[G];F=len(C)
	if F<5:return B,[_A]*X,0
	if K is _B:K=F**2
	H={B:A for(A,B)in enumerate(C)};P=[0]*F;D=[0]*F;Q=[0]*F;E=[0]*F;Y=0;Z=0
	for J in range(F):
		I=C[J];L=0;M=0
		if A.is_ppick(I):L=1
		elif A.is_pdrop(I):L=-1
		elif A.is_parc_pick(I):R=A.rev_parc_pick(I);M=A.q[R-1]
		elif A.is_parc_drop(I):R=A.rev_parc_drop(I);M=-A.q[R-1]
		Y+=L;Z+=M;P[J]=Y;D[J]=L;Q[J]=Z;E[J]=M
	S=TreeSegment(data=P,op=min,identity=float(_L),sum_like=_A);T=TreeSegment(data=P,op=max,identity=0,sum_like=_A);U=TreeSegment(data=Q,op=min,identity=float(_L),sum_like=_A);V=TreeSegment(data=Q,op=max,identity=0,sum_like=_A)
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
				if O==_G:yield(A,B,C);return
				else:yield(A,B,C)
	def g():
		A=list(f())
		if not A:return
		if O==_T:return a.choice(A)
		elif O==_U:return max(A,key=lambda x:x[2])
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
def intra_swap_operator(partial,steps=_B,mode=_G,uplift=1,seed=42,verbose=_A):
	F=verbose;C=steps;B=partial
	if C is _B:C=10**9
	D=0;G=[_A]*B.problem.K;E=B.copy()
	for A in range(B.problem.K):
		I,J,H=intra_swap_one_route_operator(E,route_idx=A,steps=C-D,mode=mode,uplift=uplift,seed=seed,verbose=F);E=I;D+=H
		if J[A]:G[A]=_C
		if F:print(f"Route {A}: performed {H} intra-route swaps.")
	return E,G,D
def cost_decrement_inter_swap(partial,route_a_idx,route_b_idx,p_idx_a,d_idx_a,p_idx_b,d_idx_b):
	M='b';L=route_b_idx;K=route_a_idx;J='a';H=partial;G=d_idx_b;F=d_idx_a;E=p_idx_b;D=p_idx_a;A=H.routes[K];B=H.routes[L];assert A[D]!=0 and B[E]!=0,_Y;T=H.route_costs[K];U=H.route_costs[L];V=H.max_cost;C=H.problem.D
	def I(from_node,routechar,idx):
		if routechar==J:D=A;E=K
		else:D=B;E=L
		if idx>=H.route_states[E][_Q]:return 0
		return C[from_node][D[idx+1]]
	if D+1==F:N=C[A[D-1]][A[D]]+C[A[D]][A[F]]+I(A[F],J,F);O=C[A[D-1]][B[E]]+C[B[E]][B[G]]+I(B[G],J,F)
	else:N=C[A[D-1]][A[D]]+C[A[D]][A[D+1]]+C[A[F-1]][A[F]]+I(A[F],J,F);O=C[A[D-1]][B[E]]+C[B[E]][A[D+1]]+C[A[F-1]][B[G]]+I(B[G],J,F)
	if E+1==G:P=C[B[E-1]][B[E]]+C[B[E]][B[G]]+I(B[G],M,G);Q=C[B[E-1]][A[D]]+C[A[D]][A[F]]+I(A[F],M,G)
	else:P=C[B[E-1]][B[E]]+C[B[E]][B[E+1]]+C[B[G-1]][B[G]]+I(B[G],M,G);Q=C[B[E-1]][A[D]]+C[A[D]][B[E+1]]+C[B[G-1]][A[F]]+I(A[F],M,G)
	R=T-N+O;S=U-P+Q;W=max(R,S,*(H.route_costs[A]for A in range(H.problem.K)if A!=K and A!=L));return R,S,V-W
def inter_swap_route_pair_operator(partial,route_a_idx,route_b_idx,steps=_B,mode=_G,uplift=1,seed=42,verbose=_A):
	P=mode;O=steps;F=route_b_idx;E=route_a_idx;c=random.Random(seed);B=partial.copy();A=B.problem;C=B.routes[E];D=B.routes[F];Q=len(C);R=len(D)
	if Q<5 or R<5:return B,[_A]*A.K,0
	def a(route):
		G=route;C=len(G);H=[0]*C;K=[0]*C;I=[0]*C;L=[0]*C;M=0;N=0
		for(D,B)in enumerate(G):
			E=0;F=0
			if A.is_ppick(B):E=1
			elif A.is_pdrop(B):E=-1
			elif A.is_parc_pick(B):J=A.rev_parc_pick(B);F=A.q[J-1]
			elif A.is_parc_drop(B):J=A.rev_parc_drop(B);F=-A.q[J-1]
			M+=E;N+=F;H[D]=M;K[D]=E;I[D]=N;L[D]=F
		O=TreeSegment(data=H,op=min,identity=float(_L),sum_like=_A);P=TreeSegment(data=H,op=max,identity=0,sum_like=_A);Q=TreeSegment(data=I,op=min,identity=float(_L),sum_like=_A);R=TreeSegment(data=I,op=max,identity=0,sum_like=_A);S={B:A for(A,B)in enumerate(G)};return S,K,L,(O,P,Q,R)
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
		T=[B for B in range(Q)if A.is_ppick(C[B])or A.is_parc_pick(C[B])];U=[B for B in range(R)if A.is_ppick(D[B])or A.is_parc_pick(D[B])]
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
				if P==_G:yield(B,E,F,G,O,S,J);return
				else:yield(B,E,F,G,O,S,J)
	def l():
		A=list(k())
		if not A:return
		if P==_T:return c.choice(A)
		elif P==_U:return max(A,key=lambda x:x[4])
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
def inter_swap_operator(partial,steps=_B,mode=_G,uplift=1,seed=42,verbose=_A):
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
def parsol_scorer(parsol,sample_size=15,w_std=.15,seed=42):
	B=parsol;D=random.Random(seed);E=max(1,sample_size);C=B.route_costs
	if len(C)==1:return B.max_cost
	A=D.choices(C,k=E);F=math.fsum(A)/len(A);G=math.fsum((A-F)**2 for A in A)/len(A);H=math.sqrt(max(_K,G));return B.max_cost+w_std*H
def beam_search_swarm_solver(problem,cost_function=parsol_scorer,initial=_B,l_width=10,r_intra=.75,r_inter=.9,f_intra=.05,f_inter=.1,verbose=_A):
	L=l_width;K=verbose;E=initial;G=cost_function;D=problem;W=time.time();F=max(1,2*(D.N+D.M)+D.K)
	def H(value):return min(max(value,_K),_H)
	X=H(r_intra);Y=H(r_inter);Z=H(f_intra);a=H(f_inter);M=max(0,int(F*X));N=max(0,int(F*Y));O=max(1,int(F*Z));P=max(1,int(F*a))
	def b(parsol):
		A=parsol;E=[];J=[A for(A,B)in enumerate(A.route_states)if not B[_D]]
		if not J:return E
		L=sorted(J,key=lambda idx:A.route_states[idx][_F]);N=min(2 if D.K>=50 else 3 if D.K>=20 else 4,len(L));O=L[:N];P=max(0,F-2*A.problem.K)
		for C in O:
			H=A.route_states[C];I=A.possible_actions(C);M=H[_E]==0 and not H[_J]and not H[_D]and H[_I]!=0
			if I:
				Q=min(1 if D.num_nodes>=500 else 2 if D.num_nodes>=200 else 4,len(I))
				for(R,S,T)in sorted(I,key=lambda item:item[2])[:Q]:A.apply_action(C,R,S,T);E.append(A.copy());A.reverse_action(C)
				if M and B>=P:A.apply_return_to_depot(C);E.append(A.copy());A.reverse_action(C)
			elif M:A.apply_return_to_depot(C);E.append(A.copy());A.reverse_action(C)
			elif K:print(f"[BeamSearch] Taxi {C} has no feasible actions and cannot return to depot.")
		E.sort(key=G);return E
	def c(parsols,use_intra,use_inter,seed_offset):
		F=seed_offset;E=use_inter;D=use_intra;C=parsols
		if not(D or E):return C
		G=[]
		for(H,I)in enumerate(C):
			A=I
			if D:A,B,B=intra_swap_operator(A,steps=_B,mode=_G,uplift=1,seed=1337+F+H,verbose=_A)
			if E:A,B,B=inter_swap_operator(A,steps=_B,mode=_G,uplift=1,seed=2671+F+H,verbose=_A)
			G.append(A)
		return G
	if E is _B:E=PartialSolutionSwarm(solutions=[PartialSolution(problem=D,routes=[])])
	A=E.parsol_list;B=E.parsol_list[0].n_actions;Q=0;assert all(A.n_actions==B for A in A),'All initial partial solutions must have the same action count.'
	while A:
		if all(A.is_complete()for A in A):break
		Q+=1;R=O is not _B and B>=M and(B-M)%O==0;S=P is not _B and B>=N and(B-N)%P==0
		if R or S:A=c(A,R,S,seed_offset=B)
		d=_A;A.sort(key=G);C=[]
		def e(cost,parsol):
			B=parsol
			if any(B.is_identical(A)for(C,A)in C):return
			A=len(C)
			while A>0 and cost<C[A-1][0]:A-=1
			C.insert(A,(cost,B))
			if len(C)>L:C.pop()
		for T in A:
			if T.n_actions!=B:continue
			for U in b(T)[:min(5,L)]:f=G(U);e(f,U)
		if not C:raise RuntimeError('Beam search stalled: no candidates generated.')
		A=[A[1]for A in C];B+=1
		if K:
			if d:print(f"[BeamSearch] Depth {B}. Diversity relaxed due to empty beam.")
			I=[A.max_cost for A in A];print(f"[BeamSearch] Depth {B}. Max_cost range: {min(I)} - {max(I)}. Avg max_cost: {sum(I)/len(I):.1f}")
	A.sort(key=G);J=PartialSolutionSwarm(solutions=A);V={_M:Q,_P:time.time()-W}
	if K:print(f"[BeamSearch] Completed. Final beam size {len(A)}");print(f"[BeamSearch] Beam max_cost range: {J.min_cost} - {J.max_cost}");print(f"[BeamSearch] Avg max_cost: {J.avg_cost}");print(f"[BeamSearch] Time taken: {V[_P]:.4f} seconds")
	return J,V
def beam_search_solver(problem,cost_function=parsol_scorer,initial=_B,l_width=10,r_intra=.75,r_inter=.9,f_intra=.05,f_inter=.1,verbose=_A):
	B=verbose;C,D=beam_search_swarm_solver(problem,cost_function,initial,l_width,r_intra,r_inter,f_intra,f_inter,B);A=C.extract_best_solution()
	if B and A:print(f"[BeamSearch] Best solution max_cost: {A.max_cost}")
	return A,D
import heapq
from dataclasses import dataclass,field
from itertools import count
from typing import Callable,Dict,List,Optional,Tuple
Action=Tuple[int,str,int,int]
ValueFunction=Callable[[PartialSolution],float]
SelectionPolicy=Callable[[PartialSolution,List[Action]],Optional[Action]]
SimulationPolicy=Callable[[PartialSolution],Optional[PartialSolution]]
DefensePolicy=Callable[[PartialSolution],Optional[Solution]]
FAILED_ROLLOUT_COST=10**12
def _enumerate_actions_greedily(partial,width):
	E=width;B=partial;assert E is not _B and E>0,'Width must be a positive integer';H=[A for(A,B)in enumerate(B.route_states)if not B[_D]]
	if not H:return[]
	I=sorted(H,key=lambda idx:B.route_states[idx][_F]);A=B.problem;K=min(2 if A.K>=50 else 3 if A.K>=20 else 4,len(I));L=I[:K];M=max(1,2*(A.N+A.M)+A.K);J=max(0,M-2*A.K);F=[]
	for D in L:
		C=B.route_states[D];G=B.possible_actions(D);N=C[_E]==0 and not C[_J]and not C[_D]and C[_I]!=0
		if G:
			O=min(1 if A.num_nodes>=500 else 2 if A.num_nodes>=200 else 4,len(G));P=sorted(G,key=lambda item:item[2])[:O]
			for(Q,R,S)in P:F.append((D,Q,R,S))
		if N and B.n_actions>=J:T=A.D[C[_I]][0];F.append((D,_V,0,T))
	U=[A for A in F if not(A[1]==_V and B.n_actions<J)];V=sorted(U,key=lambda item:item[3])[:E];return V
def _apply_action(partial,action):
	A=partial;B,C,D,E=action
	if C==_V:A.apply_return_to_depot(B)
	else:A.apply_action(B,C,D,E)
@dataclass
class RewardFunction:
	visits:int=0;min_value:float=float(_L);max_value:float=float(_S)
	def update(A,value):
		B=value
		if not math.isfinite(B):return
		A.visits+=1;A.min_value=min(A.min_value,B);A.max_value=max(A.max_value,B)
	def reward_from_value(A,value):
		B=value
		if not math.isfinite(B):return _K
		if A.visits==0:return .5
		if A.max_value==A.min_value:return .5
		C=A.max_value-A.min_value;D=(B-A.min_value)/C;return max(_K,min(_H,D))
@dataclass
class MCTSNode:
	partial:PartialSolution;parent:Optional[_Z]=_B;action:Optional[Action]=_B;width:Optional[int]=_B;children:List[_Z]=field(default_factory=list);visits:int=0;total_cost:int=0;total_reward:float=_K;untried_actions:List[Action]=field(default_factory=list)
	def __post_init__(A):A.untried_actions=_enumerate_actions_greedily(A.partial,A.width)
	@property
	def is_terminal(self):return self.partial.is_complete()
	@property
	def average_reward(self):
		A=self
		if A.visits==0:return _K
		return A.total_reward/A.visits
	@property
	def average_cost(self):
		A=self
		if A.visits==0:return _K
		return A.total_cost/A.visits
	def uct_score(A,uct_c):
		if A.visits==0:return float(_L)
		B=A.average_reward;C=A.parent.visits if A.parent else A.visits;D=uct_c*math.sqrt(math.log(C+1)/A.visits);return B+D
def _select(root,exploration):
	B=[root];A=root
	while _C:
		if A.untried_actions:return B
		if not A.children:return B
		A=max(A.children,key=lambda child:child.uct_score(exploration));B.append(A)
def _expand(node,selection_policy,width):
	C=width;A=node
	if not A.untried_actions:A.untried_actions=_enumerate_actions_greedily(A.partial,C)
	if not A.untried_actions:return
	B=selection_policy(A.partial,A.untried_actions)
	if B is _B:return
	try:A.untried_actions.remove(B)
	except ValueError:pass
	D=A.partial.copy();_apply_action(D,B);E=MCTSNode(D,parent=A,action=B,width=C);A.children.append(E);return E
def _backpropagate(path,cost,reward):
	for A in reversed(path):A.visits+=1;A.total_reward+=reward;A.total_cost+=cost
def _gather_leaves(node,value_function,limit=_B):
	A=limit
	if A is _B:A=10**6
	assert A is not _B and A>0,'Limit must be positive';B=[];G=count()
	def D(current):
		C=current
		if not C.children:
			E=value_function(C.partial);F=E,next(G),C
			if len(B)<A:heapq.heappush(B,F)
			elif E>B[0][0]:heapq.heapreplace(B,F)
			return
		for H in C.children:D(H)
	D(node);C=sorted(B,key=lambda item:item[0],reverse=_C);return[A[2]for A in C]
def _run_mcts(problem,partial,value_function,selection_policy,simulation_policy,width,uct_c,max_iters,seed,time_limit,verbose):
	Q=width;P=problem;K=time_limit;J=max_iters;I=partial;F=verbose;L=time.time();G=RewardFunction()
	if seed is not _B:random.seed(seed)
	if I is _B:I=PartialSolution(problem=P,routes=[])
	X=I or PartialSolution(problem=P,routes=[]);R=MCTSNode(X,width=Q);A=0;S=_B;B=10**9;M=0
	while _C:
		if J is not _B and A>=J:
			if F:print(f"[MCTS] Reached max iterations: {J}")
			break
		if K is not _B and time.time()-L>=K:
			if F:print(f"[MCTS] Reached time limit: {K:.2f}s")
			break
		H=_select(R,uct_c);N=H[-1];T=len(H)-1
		if T>M:M=T
		if not N.is_terminal:
			O=_expand(N,selection_policy,Q)
			if O is not _B:H.append(O);U=O
			else:U=N
		else:break
		C=simulation_policy(U.partial.copy())
		if C and C.is_complete():
			D=C.max_cost
			if D<B:B=D;S=C
			E=float(value_function(C));G.update(E);V=G.reward_from_value(E)
		else:D=FAILED_ROLLOUT_COST;E=-float(D);G.update(E);V=G.reward_from_value(E)
		_backpropagate(H,D,V);A+=1
		if F and A%1000==0:Y=time.time()-L;print(f"[MCTS] [Iteration {A}] Best rollout cost={B:.3f} MaxDepth={M} Time={Y:.2f}s")
	W={_M:A,_P:time.time()-L,'best_rollout_cost':B}
	if F:print(f"[MCTS] Iterations count={A} Time={W[_P]:.3f}s. Best rollout cost={B:.3f}")
	return R,S,W
def mcts_enumerator(problem,partial,value_function,selection_policy,simulation_policy,best_k=5,width=5,uct_c=math.sqrt(2),max_iters=500,seed=_B,time_limit=_B,verbose=_A):A=value_function;B,F,C=_run_mcts(problem,partial,width=width,uct_c=uct_c,max_iters=max_iters,value_function=A,selection_policy=selection_policy,simulation_policy=simulation_policy,time_limit=time_limit,seed=seed,verbose=verbose);D=_gather_leaves(B,value_function=A,limit=max(1,best_k));E=[A.partial.copy()for A in D];return E,C
def mcts_solver(problem,partial,value_function,selection_policy,simulation_policy,defense_policy,width=5,uct_c=math.sqrt(2),max_iters=1000,seed=_B,time_limit=_B,verbose=_A):
	D='used_best_rollout';E=time.time();F,C,A=_run_mcts(problem=problem,partial=partial,value_function=value_function,selection_policy=selection_policy,simulation_policy=simulation_policy,width=width,uct_c=uct_c,max_iters=max_iters,seed=seed,time_limit=time_limit,verbose=verbose)
	if C is _B:A[D]=_A;A['final_value']=float('nan');return _B,A
	B=C.to_solution();assert B is not _B and B.is_valid(),'Best rollout is not a valid solution.';A[D]=_C;A[_M]=A.get(_M,0);A[_P]=time.time()-E;return B,A
def read_instance():
	A,B,D=map(int,sys.stdin.readline().strip().split());E=list(map(int,sys.stdin.readline().split()));F=list(map(int,sys.stdin.readline().split()));C=[[0]*(2*A+2*B+1)for C in range(2*A+2*B+1)]
	for G in range(2*A+2*B+1):H=sys.stdin.readline().strip();C[G]=list(map(int,H.split()))
	return ShareARideProblem(A,B,D,E,F,C)
def main(verbose=_A):
	C='===============================';B=verbose;D=read_instance()
	def E(parsol):A=parsol_scorer(parsol);return-A
	def F(_ps,actions):
		A=actions;B=random.Random()
		if not A:return
		C=[float(A[3])for A in A];D=softmax_weighter(C,T=.1);E=sample_from_weight(B,D);return A[E]
	def G(ps):A,B=greedy_balanced_solver(ps.problem,premature_routes=[A.copy()for A in ps.routes],verbose=_A);return ps if A is _B else PartialSolution.from_solution(A)
	def H(ps):A,B=beam_search_solver(ps.problem,cost_function=parsol_scorer,initial=PartialSolutionSwarm([ps]));return A
	A,M=mcts_solver(problem=D,partial=_B,value_function=E,selection_policy=F,simulation_policy=G,defense_policy=H,width=3,uct_c=5.,max_iters=100000,seed=42,time_limit=2e2,verbose=B);assert A,'No solution found by MCTS.'
	if B:print();print(f"Cost after MCTS: {A.max_cost:.2f}");print(C)
	I=time.time();J=PartialSolution.from_solution(A);K,N,L=relocate_operator(partial=J,steps=_B,mode=_G,seed=111,verbose=B);A=K.to_solution();assert A,'No solution found after relocate.'
	if B:print();print(f"Total relocate performed: {L}");print(f"Cost after relocate: {A.max_cost:.2f}");print(f"Time for relocate: {time.time()-I:.2f} seconds");print(C)
	A.stdin_print(verbose=B)
if __name__=='__main__':main(verbose=_A)