_Y='MCTSNode'
_X='overtime'
_W='serveL'
_V='stochastic'
_U='best'
_T='done'
_S='dropL'
_R='pickL'
_Q='-inf'
_P='return'
_O='status'
_N='actions_done'
_M='actions'
_L='iterations'
_K='time'
_J=1.
_I='pos'
_H='first'
_G='serveP'
_F='load'
_E='parcels'
_D='ended'
_C=True
_B=False
_A=None
import sys,math,random,time,bisect,heapq
from dataclasses import dataclass,field
from typing import Any,Callable,Dict,List,Optional,Tuple,Union,Sequence,Iterator,Set
Request=Tuple[int,int,str]
SwapRequest=Tuple[int,int,str]
Action=Tuple[int,str,int,int]
RelocateRequest=Tuple[int,int,int,int,str]
CostChange=Tuple[int,int,int]
RelocateAction=Tuple[RelocateRequest,CostChange]
class ShareARideProblem:
	def __init__(A,N,M,K,parcel_qty,vehicle_caps,dist,coords=_A,name=_A):A.N=N;A.M=M;A.K=K;A.q=list(parcel_qty);A.Q=list(vehicle_caps);A.D=[A[:]for A in dist];A.num_nodes=2*N+2*M+1;A.num_requests=N+M;A.num_actions=N+2*M+K;A.num_expansions=N+2*M;A.pserve=lambda pid:(pid,pid+N+M);A.lpick=lambda lid:N+lid;A.ldrop=lambda lid:2*N+M+lid;A.rev_ppick=lambda nodeid:nodeid;A.rev_pdrop=lambda nodeid:nodeid-(N+M);A.rev_lpick=lambda n:n-N;A.rev_ldrop=lambda n:n-(2*N+M);A.is_ppick=lambda nodeid:1<=nodeid<=N;A.is_pdrop=lambda nodeid:N+M+1<=nodeid<=2*N+M;A.is_lpick=lambda nodeid:N+1<=nodeid<=N+M;A.is_ldrop=lambda nodeid:2*N+M+1<=nodeid<=2*(N+M);A.coords=coords;A.name=name
	def is_valid(A):
		if len(A.q)!=A.M:return _B
		if len(A.Q)!=A.K:return _B
		if len(A.D)!=A.num_nodes:return _B
		if not all(len(B)==A.num_nodes for B in A.D):return _B
		return _C
	def copy(A):return ShareARideProblem(A.N,A.M,A.K,A.q[:],A.Q[:],[A[:]for A in A.D],A.coords,A.name)
def route_cost_from_sequence(seq,D,verbose=_B):
	A,B=0,0
	for C in seq[1:]:B+=D[A][C];A=C
	return B
class Solution:
	def __init__(A,problem,routes,route_costs=_A):
		E=route_costs;C=routes;B=problem
		if not C:raise ValueError('Routes list cannot be empty.')
		if len(C)!=B.K:raise ValueError(f"Expected {B.K} routes, got {len(C)}.")
		if not E:D=[route_cost_from_sequence(A,B.D)for A in C]
		else:D=E
		A.problem=B;A.routes=C;A.route_costs=D;A.num_actions=B.num_actions;A.max_cost=max(D)if D else 0
	def is_valid(G):
		A=G.problem;J=A.K
		if len(G.routes)!=J:return _B
		H=set()
		for(K,D)in enumerate(G.routes):
			if not(D[0]==0 and D[-1]==0):return _B
			L=len(D);E=0;F=set()
			for(I,B)in enumerate(D[1:-1],start=1):
				if B in H:return _B
				H.add(B)
				if A.is_ppick(B):
					M=A.rev_ppick(B);N=A.pserve(M)[1]
					if I+1>=L or D[I+1]!=N:return _B
				elif A.is_pdrop(B):0
				elif A.is_lpick(B):
					C=A.rev_lpick(B)
					if C in F:return _B
					E+=A.q[C-1]
					if E>A.Q[K]:return _B
					F.add(C)
				elif A.is_ldrop(B):
					C=A.rev_ldrop(B)
					if C not in F:return _B
					if E-A.q[C-1]<0:return _B
					E-=A.q[C-1];F.remove(C)
			if E!=0:return _B
		if len(H)!=A.num_requests*2:return _B
		return _C
	def stdin_print(A,verbose=_B):
		print(A.problem.K)
		for(B,C)in zip(A.routes,A.route_costs):print(len(B));print(' '.join(map(str,B)))
class PartialSolution:
	def __init__(A,problem,routes=_A):C=routes;B=problem;A.problem=B;A.routes=A._init_routes(C);A.route_costs=A._init_costs(C);A.min_cost=min(A.route_costs);A.max_cost=max(A.route_costs);A.avg_cost=sum(A.route_costs)/B.K;A.node_assignment=A._init_node_assignment();A.remaining_pass_serve,A.remaining_parc_pick,A.remaining_parc_drop,A.states,A.num_actions=A._init_states()
	def _init_routes(D,routes=_A):
		A=routes;B=D.problem.K
		if not A:return[[0]for A in range(B)]
		if len(A)!=B:raise ValueError(f"Expected {B} routes, got {len(A)}.")
		for C in A:
			if not C:raise ValueError('One route cannot be null')
			elif C[0]!=0:raise ValueError('Each route must start at depot 0.')
		return A
	def _init_costs(A,routes=_A):
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
			Q=E[-1];R=M>1 and E[-1]==0;S={_I:Q,_E:F,_F:H,_M:D,_D:R};L.append(S)
		return J,K,G,L,D
	def is_valid(B,verbose=_B):
		A=B.problem;W,X,Q=A.N,A.M,A.K
		if not len(B.routes)==len(B.states)==len(B.route_costs)==Q:return _B
		if len(B.node_assignment)!=len(A.D):return _B
		R=set(range(1,W+1));S=set(range(1,X+1));L=set();M=[-1]*len(A.D);I=0;N=0;Y=0
		for F in range(Q):
			E=B.routes[F];J=len(E);O=B.states[F]
			if not E or E[0]!=0:return _B
			T=J>1 and E[-1]==0
			if O[_D]!=T:return _B
			G=set();J=len(E);H=0;U=E[0];K=0
			for(P,C)in enumerate(E[1:],start=1):
				if not 0<=C<A.num_nodes:return _B
				if C!=0:
					Z=M[C]
					if Z not in(-1,F):return _B
					M[C]=F
				K+=A.D[U][C];U=C
				if A.is_ppick(C):
					V=A.rev_ppick(C)
					if P+1<J:
						a=E[P+1];b=A.pserve(V)[1]
						if a!=b:return _B
					I+=1;R.discard(V)
				elif A.is_pdrop(C):0
				elif A.is_lpick(C):
					D=A.rev_lpick(C)
					if D in G:return _B
					H+=A.q[D-1]
					if H>A.Q[F]:return _B
					I+=1;G.add(D);S.discard(D);L.add(D)
				elif A.is_ldrop(C):
					D=A.rev_ldrop(C)
					if D not in G:return _B
					H-=A.q[D-1]
					if H<0:return _B
					I+=1;G.remove(D);L.discard(D)
				else:
					if P!=J-1:return _B
					if H!=0 or G:return _B
					if not T:return _B
					I+=1
			if O[_E]!=G:return _B
			if O[_F]!=H:return _B
			if B.route_costs[F]!=K:return _B
			N=max(N,K);Y+=K
		if R!=B.remaining_pass_serve:return _B
		if S!=B.remaining_parc_pick:return _B
		if L!=B.remaining_parc_drop:return _B
		if M!=B.node_assignment:return _B
		if B.max_cost!=N:return _B
		if B.num_actions!=I:return _B
		return _C
	def is_pending(A):return A.num_actions<A.problem.num_actions
	def is_identical(A,other):
		B=other
		if A is B:return _C
		if A.problem is not B.problem or A.num_actions!=B.num_actions:return _B
		return sorted(tuple(A[:3])for A in A.routes)==sorted(tuple(A[:3])for A in B.routes)
	def copy(A):return PartialSolution(problem=A.problem,routes=[A.copy()for A in A.routes]if A.routes else _A)
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
		A=E.problem;G=F[_I];C=[]
		for I in E.remaining_pass_serve:J,K=A.pserve(I);D=A.D[G][J]+A.D[J][K];C.append((_G,I,D))
		for B in E.remaining_parc_pick:
			L=A.q[B-1]
			if F[_F]+L<=A.Q[H]:D=A.D[G][A.lpick(B)];C.append((_R,B,D))
		for B in F[_E]:D=A.D[G][A.ldrop(B)];C.append((_S,B,D))
		C.sort(key=lambda x:x[2]);return C
	def check_expand(A,route_idx,kind,actid):
		E=route_idx;C=actid;B=kind;D=A.states[E];F=A.problem
		if D[_D]:return _B
		if B==_G:return C in A.remaining_pass_serve
		if B==_R:return C in A.remaining_parc_pick and D[_F]+F.q[C-1]<=F.Q[E]
		if B==_S:return C in D[_E]
		raise ValueError(f"Unknown action kind: {B}")
	def check_return(B,route_idx):A=B.states[route_idx];return not(A[_D]or A[_E])
	def apply_extend(A,route_idx,kind,actid,inc):
		G=kind;C=actid;B=route_idx;I=A.routes[B];D=A.states[B];E=A.problem
		if D[_D]:raise ValueError(f"Cannot apply action on ended route {B}.")
		if G==_G:K,J=E.pserve(C);I.append(K);I.append(J);A.node_assignment[K]=B;A.node_assignment[J]=B;A.remaining_pass_serve.discard(C);D[_I]=J;D[_M]+=1;A.route_costs[B]+=inc;A.max_cost=max(A.max_cost,A.route_costs[B]);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions+=1;return
		elif G==_R:
			F=E.q[C-1]
			if D[_F]+F>E.Q[B]:raise ValueError(f"Taxi {B} capacity exceeded for parcel {C}.")
			H=E.lpick(C);D[_F]+=F;D[_E].add(C);A.remaining_parc_pick.discard(C);A.remaining_parc_drop.add(C)
		elif G==_S:
			F=E.q[C-1]
			if D[_F]-F<0:raise ValueError(f"Taxi {B} load cannot be negative after dropping parcel {C}.")
			H=E.ldrop(C);D[_F]-=F;D[_E].discard(C);A.remaining_parc_drop.discard(C)
		else:raise ValueError(f"Unknown action kind: {G}")
		D[_I]=H;D[_M]+=1;I.append(H);A.node_assignment[H]=B;A.route_costs[B]+=inc;A.max_cost=max(A.max_cost,A.route_costs[B]);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions+=1
	def apply_return(A,t_idx):
		C=t_idx;D=A.routes[C];B=A.states[C]
		if B[_D]:return
		if B[_E]:raise ValueError(f"Taxi {C} must drop all loads before returning to depot.")
		E=A.problem.D[B[_I]][0];D.append(0);B[_I]=0;B[_M]+=1;B[_D]=_C;A.route_costs[C]+=E;A.max_cost=max(A.max_cost,A.route_costs[C]);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions+=1
	def reverse_action(A,t_idx):
		G=t_idx;F=A.routes[G];C=A.states[G]
		if len(F)<=1:raise ValueError(f"No actions to reverse for taxi {G}.")
		B=A.problem;D=F[-1]
		if B.is_pdrop(D):
			J=F.pop();I=F.pop();L=B.rev_pdrop(J)
			if B.rev_ppick(I)!=L:raise ValueError('Inconsistent route state: pdrop not preceded by corresponding ppick.')
			H=F[-1];K=B.D[H][I]+B.D[I][J];C[_I]=H;C[_M]-=1;C[_D]=_B;A.remaining_pass_serve.add(L);A.node_assignment[J]=-1;A.node_assignment[I]=-1;A.route_costs[G]-=K;A.max_cost=max(A.route_costs);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions-=1;return
		D=F.pop();H=F[-1];K=B.D[H][D];C[_I]=H;C[_M]-=1;C[_D]=_B
		if B.is_lpick(D):E=B.rev_lpick(D);C[_F]-=B.q[E-1];C[_E].discard(E);A.remaining_parc_pick.add(E);A.remaining_parc_drop.discard(E)
		elif B.is_ldrop(D):E=B.rev_ldrop(D);C[_F]+=B.q[E-1];C[_E].add(E);A.remaining_parc_pick.discard(E);A.remaining_parc_drop.add(E)
		elif D==0:0
		else:raise ValueError(f"Unexpected node type to reverse: {D}")
		A.route_costs[G]-=K;A.max_cost=max(A.route_costs);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.node_assignment[D]=-1;A.num_actions-=1
	def is_completed(A,verbose=_B):
		if A.num_actions!=A.problem.num_actions:return _B
		if not all(A[_D]for A in A.states):return _B
		return _C
	def to_solution(A):
		if not A.is_completed(verbose=_C):return
		if not A.is_valid(verbose=_C):return
		B=Solution(problem=A.problem,routes=A.routes,route_costs=A.route_costs);return B
	@staticmethod
	def from_solution(sol):A=[A.copy()for A in sol.routes];return PartialSolution(problem=sol.problem,routes=A)
class PartialSolutionSwarm:
	def __init__(A,solutions):
		B=solutions
		if not B:raise ValueError('Solutions list cannot be empty.')
		A.problem=B[0].problem;A.num_partials=len(B);A.partial_lists=B;A.partial_num_actions=[A.num_actions for A in B];A.partial_costs=[A.max_cost for A in B];A.min_cost=min(A.partial_costs);A.max_cost=max(A.partial_costs);A.avg_cost=sum(A.max_cost for A in B)/len(B)
	def update(A):A.partial_num_actions=[A.num_actions for A in A.partial_lists];A.partial_costs=[A.max_cost for A in A.partial_lists];A.min_cost=min(A.partial_costs);A.max_cost=max(A.partial_costs);A.avg_cost=sum(A.max_cost for A in A.partial_lists)/len(A.partial_lists)
	def opt(B):
		B.update();C=10**18;D=_A
		for E in B.partial_lists:
			if E.is_completed():
				A=E.to_solution()
				if A and A.max_cost<C:C=A.max_cost;D=A
		return D
	def stats(A):A.update();return{'num_partials':A.num_partials,'min_cost':A.min_cost,'max_cost':A.max_cost,'avg_cost':A.avg_cost}
	def copy(A):B=[A.copy()for A in A.partial_lists];return PartialSolutionSwarm(solutions=B)
def weighted(kind,inc,pweight=.7):
	if kind==_G:return pweight*inc
	return inc+1
def action_weight(action,pweight=.7):A=action;return weighted(A[1],A[3],pweight)
def softmax_weighter(incs,t):
	A=incs;B,E=min(A),max(A);C=E-B
	if C<1e-06:return[_J]*len(A)
	D=[]
	for F in A:G=(F-B)/C;D.append((_J-G+.1)**(_J/t))
	return D
def balanced_scorer(partial,sample_size=8,w_std=.15,seed=_A):
	B=sample_size;A=partial;E=random.Random(seed);C=sorted(A.route_costs)
	if len(C)==1:return A.max_cost
	D=E.choices(C,k=B);F=sum(D)/B;G=sum((A-F)**2 for A in D)/B;H=G**.5;return A.max_cost+w_std*H
def check_general_action(partial,action):
	A=partial;B,C,D,E=action
	if C==_P:return A.check_return(B)
	return A.check_expand(B,C,D)
def apply_general_action(partial,action):
	A=partial;B,C,D,E=action
	if C==_P:A.apply_return(B)
	else:A.apply_extend(B,C,D,E)
def enumerate_actions_greedily(partial,width=_A,asymmetric=_C):
	D=width;A=partial
	if D is _A:D=10**9
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
	if not F:F=N(aggressive=_B)
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
			if A.check_return(C):Z=B.D[Y[_I]][0];S.append((C,_P,0,Z))
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
def repair_one_route(partial,route_idx,steps,T=_J,seed=_A,verbose=_B):
	B=route_idx;A=partial;E=random.Random(seed);C=0
	for M in range(steps):
		F=A.states[B]
		if F[_D]:break
		D=A.possible_expand(B)
		if not D:A.apply_return(B);C+=1;break
		G=[weighted(A,B)for(A,C,B)in D];H=softmax_weighter(G,T);I=sample_from_weight(E,H);J,K,L=D[I];A.apply_extend(B,J,K,L);C+=1
	return A,C
def repair_operator(partial,repair_proba=_A,steps=_A,T=_J,seed=_A,verbose=_B):
	C=steps;B=repair_proba;A=partial;E=random.Random(seed)
	if B is _A:B=_J
	if C is _A:C=10**9
	I=list(range(A.problem.K));D=A.problem.K;J=round(B*D+.5);K=min(D,max(1,J));L=E.sample(I,K);F=0;G=[_B]*D
	for H in L:A,M=repair_one_route(partial=A,route_idx=H,steps=C,T=T,seed=E.randint(0,1000000),verbose=verbose);F+=M;G[H]=_C
	return A,G,F
def destroy_one_route(problem,route,route_idx,steps=10,verbose=_B):
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
def destroy_operator(sol,destroy_proba,destroy_steps,seed=_A,t=_J,verbose=_B):
	A=sol;M=random.Random(seed);B=A.problem.K;D=[A[:]for A in A.routes];N=A.route_costs;O=round(destroy_proba*B+.5);P=min(B,max(1,O));Q=softmax_weighter(N,t=t);G=[];E=list(range(B));H=Q[:]
	for T in range(P):
		if not E:break
		F=sample_from_weight(M,H);G.append(E[F]);E.pop(F);H.pop(F)
	I=[_B]*B;J=0
	for C in G:
		K=D[C]
		if len(K)<=2:continue
		R,L=destroy_one_route(A.problem,K,C,steps=destroy_steps,verbose=verbose)
		if L>0:D[C]=R;I[C]=_C;J+=L
	S=PartialSolution(problem=A.problem,routes=D);return S,I,J
def greedy_solver(problem,partial=_A,num_actions=7,t_actions=.01,seed=_A,verbose=_B):
	A=partial;D=time.time();G=random.Random(seed)
	if A is _A:A=PartialSolution(problem=problem)
	C=0;E=A.num_actions
	while A.is_pending():
		C+=1;F=enumerate_actions_greedily(A,num_actions);B=[A for A in F if A[1]!=_P]
		if not B:B=F
		if not B:return _A,{_L:C,_K:time.time()-D,_N:A.num_actions-E,_O:'error'}
		H=[A[3]for A in B];I=softmax_weighter(H,t_actions);J=sample_from_weight(G,I);K=B[J];apply_general_action(A,K)
	L=A.to_solution();M={_L:C,_K:time.time()-D,_N:A.num_actions-E,_O:_T};return L,M
def iterative_greedy_solver(problem,partial=_A,iterations=10000,num_actions=7,t_actions=.01,destroy_proba=.53,destroy_steps=13,destroy_t=1.3,rebuild_proba=.29,rebuild_steps=3,rebuild_t=1.2,time_limit=3e1,seed=_A,verbose=_B):
	N=rebuild_steps;M=rebuild_proba;L=destroy_steps;K=destroy_proba;F=partial;E=problem;B=seed;G=time.time();O=G+time_limit;X=random.Random(B);assert 1e-05<K<.99999;assert 1e-05<M<.99999;assert 1<=N<=L
	if F is _A:F=PartialSolution(problem=E,routes=[])
	C,Y=greedy_solver(problem=E,partial=F,num_actions=num_actions,t_actions=t_actions,seed=3*B if B else _A,verbose=verbose)
	if not C:return _A,{_K:time.time()-G,_O:'error'}
	H=C.max_cost;D=Y[_N];P=H;Q=0;R=0;S=0;T=_T;I=0
	for Z in range(1,iterations+1):
		if O and time.time()>=O:T=_X;break
		U=_A if B is _A else 2*B+98*Z;a,b,V=destroy_operator(C,K,L,seed=U,t=destroy_t);R+=V;D+=V;J=a
		for(c,d)in enumerate(b):
			if not d:continue
			if X.random()>M:continue
			J,W=repair_one_route(J,route_idx=c,steps=N,T=rebuild_t,seed=U);S+=W;D+=W
		A,e=greedy_solver(E,partial=J,num_actions=1,verbose=_B);P+=A.max_cost if A else 0;D+=e[_N];I+=1 if A else 0
		if A and A.max_cost<H:C=A;H=A.max_cost;Q+=1
	f=time.time()-G;g={_L:I,_N:D,'improvements':Q,'actions_destroyed':R,'actions_rebuilt':S,'average_cost':P/(I+1),_K:f,_O:T};return C,g
class TreeSegment:
	def __init__(A,data,op,identity,sum_like=_C,add_neutral=0):
		A.num_elements=len(data);A.op=op;A.identity=identity;A.sum_like=sum_like;A.num_leaves=1
		while A.num_leaves<A.num_elements:A.num_leaves*=2
		A.data=[A.identity]*(2*A.num_leaves);A.lazy=[add_neutral]*(2*A.num_leaves)
		for B in range(A.num_elements):A.data[A.num_leaves+B]=data[B]
		for B in range(A.num_leaves-1,0,-1):A.data[B]=A.op(A.data[2*B],A.data[2*B+1])
	def _apply(A,x,val,length):
		B=val
		if A.sum_like:A.data[x]+=B*length
		else:A.data[x]+=B
		if x<A.num_leaves:A.lazy[x]+=B
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
	def update(A,l,r,val):A._update(l,r,val,1,0,A.num_leaves)
	def query(A,l,r):return A._query(l,r,1,0,A.num_leaves)
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
	def query_max_prefix(I,l,r):
		B=float(_Q);A=0;C=0;B=float(_Q);A=0
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
def cost_decrement_intra_swap(partial,route_idx,a_idx,len_a,b_idx,len_b):
	P=partial;I=len_b;H=len_a;C=b_idx;B=a_idx
	if B>C:B,C=C,B;H,I=I,H
	A=P.problem.D;D=P.routes[route_idx];Q=len(D);J=D[B-1];K=D[B];L=D[B+H-1];M=D[B+H]if B+H<Q else _A;R=D[C-1];N=D[C];O=D[C+I-1];E=D[C+I]if C+I<Q else _A;F=0;G=0
	if B+H==C:
		F+=A[J][K];F+=A[L][N]
		if E is not _A:F+=A[O][E]
		G+=A[J][N];G+=A[O][K]
		if E is not _A:G+=A[L][E]
	else:
		F+=A[J][K]
		if M is not _A:F+=A[L][M]
		F+=A[R][N]
		if E is not _A:F+=A[O][E]
		G+=A[J][N]
		if M is not _A:G+=A[O][M]
		G+=A[R][K]
		if E is not _A:G+=A[L][E]
	return F-G
def intra_swap_one_route_operator(partial,route_idx,steps=_A,mode=_H,uplift=1,seed=_A,verbose=_B):
	M=mode;L=steps;K=route_idx;D=partial;a=random.Random(seed);D=D.copy();C=D.problem;F=D.routes[K];R=C.K;N=len(F)
	if N<5:return D,[_B]*R,0
	if L is _A:L=N**2
	H={};A=[];E=[];I={}
	def b():
		nonlocal H,A,E,I;H={B:A for(A,B)in enumerate(F)};B=1;L=0
		while B<N-1:
			G=F[B];D=0;J=0
			if C.is_ppick(G):
				if B+2>N-1:B+=1;continue
				D=2;J=0
			elif C.is_lpick(G):D=1;K=C.rev_lpick(G);J=C.q[K-1]
			elif C.is_ldrop(G):D=1;K=C.rev_ldrop(G);J=-C.q[K-1]
			else:B+=1;continue
			A.append([B,D]);E.append(J)
			for M in range(D):I[B+M]=L
			B+=D;L+=1
	b();G=len(A);B=[-1]*G
	for J in range(G):
		c,l=A[J];S=F[c]
		if C.is_lpick(S):
			d=C.rev_lpick(S);T=C.ldrop(d)
			if T in H:
				U=H[T]
				if U in I:V=I[U];B[J]=V;B[V]=J
	O=[0]*G;W=0
	for J in range(G):W+=E[J];O[J]=W
	X=TreeSegment(O,min,10**18,_B);Y=TreeSegment(O,max,0,_B)
	def e(i,j):
		if E[i]>0 and B[i]<=j:return _B
		if E[j]<0 and B[j]>=i:return _B
		return _C
	def f(i,j):
		B=E[i];D=E[j];A=D-B
		if A>0:
			if Y.query(i,j)+A>C.Q[K]:return _B
		elif A<0:
			if X.query(i,j)+A<0:return _B
		return _C
	def g(i,j):
		if not(e(i,j)and f(i,j)):return _B,0
		B,C=A[i];E,F=A[j];G=cost_decrement_intra_swap(D,K,B,C,E,F);return _C,G
	def h():
		for A in range(G):
			for B in range(A+1,G):
				D,C=g(A,B)
				if not D or C<uplift:continue
				yield(A,B,C)
				if M==_H:return
	def i():
		A=list(h())
		if not A:return
		if M==_H:return A[0]
		elif M==_U:return max(A,key=lambda x:x[2])
		elif M==_V:return a.choice(A)
	P=0;Z=[_B]*R
	def j(action):nonlocal F,D;H,I,J=action;B,E=A[H];C,G=A[I];L=F[:B]+F[C:C+G]+F[B+E:C]+F[B:B+E]+F[C+G:];F[:]=L;D.decrease_cost(K,J)
	def k(action):
		nonlocal E,B,A,H,I;C,D,Q=action;J=E[D]-E[C]
		if J:X.update(C,D,J);Y.update(C,D,J)
		E[C],E[D]=E[D],E[C]
		if B[C]!=-1:B[B[C]]=D
		if B[D]!=-1:B[B[D]]=C
		B[C],B[D]=B[D],B[C];L,M=A[C];N,O=A[D];K=O-M;A[C],A[D]=A[D],A[C];A[C][0],A[D][0]=L,N+K
		if K:
			for P in range(C+1,D):A[P][0]+=K
		H={B:A for(A,B)in enumerate(F)};I={A[B][0]+C:B for B in range(G)for C in range(A[B][1])}
	while _C:
		if L is not _A and P>=L:break
		Q=i()
		if Q is _A:break
		j(Q);k(Q);P+=1;Z[K]=_C
	return D,Z,P
def intra_swap_operator(partial,steps=_A,mode=_H,uplift=1,seed=_A,verbose=_B):
	F=verbose;E=partial;B=steps
	if B is _A:B=10**9
	C=0;G=E.problem.K;H=[_B]*G;A=E.copy()
	for D in range(G):
		I,J,K=intra_swap_one_route_operator(A,route_idx=D,steps=B-C,mode=mode,uplift=uplift,seed=seed,verbose=F);A=I;C+=K
		if J[D]:H[D]=_C
	if A.is_valid(verbose=F)is _B:A.stdin_print();raise ValueError('Intra-swap operator produced invalid solution.')
	return A,H,C
def cost_decrement_inter_swap(partial,raidx,rbidx,paidx,qaidx,pbidx,qbidx):
	T=rbidx;S=raidx;P=qbidx;O=pbidx;N=qaidx;M=paidx;F=partial;A=F.problem.D;G=F.routes[S];H=F.routes[T];a=len(G);b=len(H);assert G[M]!=0 and H[O]!=0,'Cannot swap depot nodes.';c=F.route_costs[S];d=F.route_costs[T];e=F.max_cost;Q=G[M-1];I=G[M];U=G[M+1];V=G[N-1];J=G[N];B=_A
	if N+1<a:B=G[N+1]
	R=H[O-1];K=H[O];W=H[O+1];X=H[P-1];L=H[P];C=_A
	if P+1<b:C=H[P+1]
	D=0
	if M+1==N:
		D-=A[Q][I]+A[I][J]
		if B is not _A:D-=A[J][B]
		D+=A[Q][K]+A[K][L]
		if B is not _A:D+=A[L][B]
	else:
		D-=A[Q][I]+A[I][U]+A[V][J]
		if B is not _A:D-=A[J][B]
		D+=A[Q][K]+A[K][U]+A[V][L]
		if B is not _A:D+=A[L][B]
	E=0
	if O+1==P:
		E-=A[R][K]+A[K][L]
		if C is not _A:E-=A[L][C]
		E+=A[R][I]+A[I][J]
		if C is not _A:E+=A[J][C]
	else:
		E-=A[R][K]+A[K][W]+A[X][L]
		if C is not _A:E-=A[L][C]
		E+=A[R][I]+A[I][W]+A[X][J]
		if C is not _A:E+=A[J][C]
	Y=c+D;Z=d+E;f=[F.route_costs[A]for A in range(F.problem.K)if A!=S and A!=T];g=max(Y,Z,*f);h=e-g;return Y,Z,h
def inter_swap_route_pair_operator(partial,route_a_idx,route_b_idx,steps=_A,mode=_H,uplift=1,seed=_A,verbose=_B):
	N=mode;M=steps;H=route_b_idx;G=route_a_idx;W=random.Random(seed);B=partial.copy();A=B.problem;C=B.routes[G];D=B.routes[H];S=len(C);T=len(D)
	if S<5 or T<5:return B,[_B]*A.K,0
	def U(route):
		D=route;G=len(D);E=[0]*G;H=[0]*G;I=0
		for(J,B)in enumerate(D):
			C=0
			if A.is_ppick(B)or A.is_pdrop(B):0
			elif A.is_lpick(B):F=A.rev_lpick(B);C=A.q[F-1]
			elif A.is_ldrop(B):F=A.rev_ldrop(B);C=-A.q[F-1]
			I+=C;E[J]=I;H[J]=C
		K=TreeSegment(data=E,op=min,identity=10**18,sum_like=_B);L=TreeSegment(data=E,op=max,identity=0,sum_like=_B);M={B:A for(A,B)in enumerate(D)};return M,H,K,L
	I,E,O,P=U(C);J,F,Q,R=U(D);X=A.Q[G];Y=A.Q[H]
	def Z(req_a,req_b):
		A,G,M=req_a;B,H,N=req_b;C=F[B]-E[A];D=E[A]-F[B]
		if C!=0:
			I=O.query(A,G);J=P.query(A,G)
			if I+C<0 or J+C>X:return _B
		if D!=0:
			K=Q.query(B,H);L=R.query(B,H)
			if K+D<0 or L+D>Y:return _B
		return _C
	def a(req_a,req_b):
		A,B,C=req_a;D,E,F=req_b
		if C==_G:
			if E!=D+1:return _B
		if F==_G:
			if B!=A+1:return _B
		return _C
	def b(req_a,req_b):
		C=req_b;A=req_a;D,E,F=A;I,J,F=C
		if not a(A,C):return _B,0,0,0
		if not Z(A,C):return _B,0,0,0
		K,L,M=cost_decrement_inter_swap(B,G,H,D,E,I,J);return _C,K,L,M
	def c():
		H=[B for B in range(S)if A.is_ppick(C[B])or A.is_lpick(C[B])];K=[B for B in range(T)if A.is_ppick(D[B])or A.is_lpick(D[B])]
		def F(p_idx,route,pos):
			B=p_idx;D=route[B];C=''
			if A.is_ppick(D):E=B+1;C=_G
			else:
				G=A.rev_lpick(D);F=A.ldrop(G)
				if F not in pos:return
				E=pos[F];C=_W
			return B,E,C
		for L in H:
			B=F(L,C,I)
			if B is _A:continue
			for M in K:
				E=F(M,D,J)
				if E is _A:continue
				O,P,Q,G=b(B,E)
				if not O or G<uplift:continue
				yield(B,E,P,Q,G)
				if N==_H:return
	def d():
		A=list(c())
		if not A:return
		if N==_V:return W.choice(A)
		elif N==_U:return max(A,key=lambda x:x[4])
		else:return A[0]
	K=0;V=0;L=[_B]*A.K
	if M is _A:M=10**9
	def e(action):nonlocal C,D,B;nonlocal I,J;P,Q,R,S,T=action;L,M,U=P;N,O,U=Q;A,E=C[L],C[M];F,K=D[N],D[O];del I[A];del I[E];I[F]=L;I[K]=M;del J[F];del J[K];J[A]=N;J[E]=O;C[L],C[M]=F,K;D[N],D[O]=A,E;B.node_assignment[A]=H;B.node_assignment[E]=H;B.node_assignment[F]=G;B.node_assignment[K]=G;B.route_costs[G]=R;B.route_costs[H]=S;B.max_cost-=T
	def f(action):
		nonlocal I,J;nonlocal E,F;nonlocal O,P;nonlocal Q,R;L,M,K,__,___=action;A,C,K=L;B,D,K=M;G=F[B]-E[A]
		if G!=0:O.update(A,C,G);P.update(A,C,G)
		H=E[A]-F[B]
		if H!=0:Q.update(B,D,H);R.update(B,D,H)
		E[A],F[B]=F[B],E[A];E[C],F[D]=F[D],E[C]
	def g():
		nonlocal K,L,V
		while K<M:
			A=d()
			if A is _A:break
			f(A);e(A);V+=A[4];L[G]=_C;L[H]=_C;K+=1
	g();return B,L,K
def inter_swap_operator(partial,steps=_A,mode=_H,uplift=1,seed=_A,verbose=_B):
	M=verbose;H=steps;G=partial;T=random.Random(seed);I=G.problem.K
	if I<2:return G.copy(),[_B]*I,0
	A=G.copy();J=[_B]*I;D=0;N=H if H is not _A else 10**9;E=[(-B,A)for(A,B)in enumerate(A.route_costs)];C=[(B,A)for(A,B)in enumerate(A.route_costs)];heapq.heapify(E);heapq.heapify(C)
	def U():
		while E:
			B,C=heapq.heappop(E)
			if-B==A.route_costs[C]:return-B,C
	def V(exclude_idx=_A):
		while C:
			D,B=heapq.heappop(C)
			if B==exclude_idx:continue
			if D==A.route_costs[B]:return D,B
	def K(idx):B=idx;D=A.route_costs[B];heapq.heappush(E,(-D,B));heapq.heappush(C,(D,B))
	while _C:
		if H is not _A and D>=N:break
		O=U()
		if O is _A:break
		W,B=O;P=[];Q=_B
		while _C:
			L=V(exclude_idx=B)
			if L is _A:break
			W,F=L;P.append(L);X,R,S=inter_swap_route_pair_operator(A,route_a_idx=B,route_b_idx=F,steps=N-D,mode=mode,uplift=uplift,seed=T.randint(10,10**9),verbose=M)
			if S>0:
				K(B);K(F);A=X;D+=S
				if R[B]:J[B]=_C
				if R[F]:J[F]=_C
				Q=_C;break
		for(Y,Z)in P:heapq.heappush(C,(Y,Z))
		if not Q:K(B);break
	if A.is_valid(verbose=M)is _B:raise ValueError('Inter-swap operator produced invalid solution.')
	return A,J,D
def cost_decrement_relocate(partial,rfidx,rtidx,pfidx,qfidx,ptidx,qtidx):
	R=qtidx;Q=ptidx;P=rtidx;O=rfidx;I=qfidx;H=pfidx;B=partial;A=B.problem.D;Y=B.max_cost;E=B.routes[O];J=B.routes[P];Z=B.route_costs[O];a=B.route_costs[P];C=E[H];D=E[I];K=E[H-1];S=E[H+1];T=E[I-1];L=E[I+1];F=0
	if H+1==I:F-=A[K][C]+A[C][D]+A[D][L];F+=A[K][L]
	else:F-=A[K][C]+A[C][S]+A[T][D]+A[D][L];F+=A[K][S]+A[T][L]
	U=Z+F;M=J[Q-1];V=J[Q];W=J[R-2];N=J[R-1];G=0
	if R==Q+1:G-=A[M][N];G+=A[M][C]+A[C][D]+A[D][N]
	else:G-=A[M][V]+A[W][N];G+=A[M][C]+A[C][V]+A[W][D]+A[D][N]
	X=a+G;b=[B.route_costs[A]for A in range(B.problem.K)if A!=O and A!=P];c=max(U,X,*b);return U,X,Y-c
def relocate_from_to(partial,route_from_idx,route_to_idx,steps,mode,uplift=1,seed=_A,verbose=_B):
	L=mode;K=partial;F=route_from_idx;E=route_to_idx;N=random.Random(seed);A=K.problem;C=K.copy();B=C.routes[F];D=C.routes[E];H=len(B);G=len(D)
	if H<5:return K,[_B]*A.K,0
	def M(route,n):
		E=[0]*n
		for(F,B)in enumerate(route):
			if A.is_lpick(B):C=A.rev_lpick(B);D=A.q[C-1]
			elif A.is_ldrop(B):C=A.rev_ldrop(B);D=-A.q[C-1]
			else:D=0
			E[F]=D
		G=MinMaxPfsumArray(E);return G
	I=M(B,H);J=M(D,G);O=A.Q[F];P=A.Q[E]
	def Q(req):
		D,E,A,B,C=req
		if C==_W:return _C
		return B==A+1
	def R(req):
		D,E,F,G,K=req
		if K==_G:return _C
		H=B[D]
		if A.is_lpick(H):L=A.rev_lpick(H);C=A.q[L-1]
		else:C=0
		M=I.query_min_prefix(D,E);N=I.query_max_prefix(D,E)
		if M-C<0:return _B
		if N-C>O:return _B
		Q=J.query_min_prefix(F-1,G-1);R=J.query_max_prefix(F-1,G-1)
		if Q+C<0:return _B
		if R+C>P:return _B
		return _C
	def S(req):
		A=req
		if not Q(A):return
		if not R(A):return
		B=cost_decrement_relocate(C,F,E,A[0],A[1],A[2],A[3]);return B
	def T():
		M={B:A for(A,B)in enumerate(B)};H=[]
		for(E,I)in enumerate(B[1:],start=1):
			if A.is_ppick(I):C=E+1;H.append((E,C,_G))
			elif A.is_lpick(I):
				N=A.rev_lpick(I);O=A.ldrop(N);C=M.get(O)
				if C is not _A and C>E:H.append((E,C,_W))
		P=[(B,B+1)for B in range(1,G)if not A.is_ppick(D[B-1])];Q=[(B,C)for B in range(1,G)if not A.is_ppick(D[B-1])for C in range(B+1,G+1)if not A.is_ppick(D[C-2])]
		for(E,C,K)in H:
			R=P if K==_G else Q
			for(T,U)in R:
				J=E,C,T,U,K;F=S(J)
				if F is _A:continue
				V,V,W=F
				if W<uplift:continue
				if L==_H:yield(J,F);return
				else:yield(J,F)
	def U():
		A=list(T())
		if not A:return
		if L==_V:return N.choice(A)
		elif L==_U:return max(A,key=lambda x:x[1][2])
		else:return A[0]
	def V(action):nonlocal B,D,C;(A,G,J,K,O),(L,M,N)=action;H=B[A];I=B[G];del B[G];del B[A];D.insert(J,H);D.insert(K,I);C.routes[F]=B;C.routes[E]=D;C.route_costs[F]=L;C.route_costs[E]=M;C.max_cost-=N;C.node_assignment[H]=E;C.node_assignment[I]=E
	def W(action):
		nonlocal I,J,B,D;(C,E,G,H,M),N=action;K=B[C];L=B[E]
		def F(nodeid):
			B=nodeid
			if A.is_lpick(B):C=A.rev_lpick(B);return A.q[C-1]
			elif A.is_ldrop(B):C=A.rev_ldrop(B);return-A.q[C-1]
			else:return 0
		I.delete(E);I.delete(C);J.insert(G,F(K));J.insert(H,F(L))
	def X():
		nonlocal H,G,B,D;C=0;I=[_B]*A.K
		while C<steps:
			J=U()
			if J is _A:break
			W(J);V(J);C+=1;I[F]=_C;I[E]=_C;H-=2;G+=2
			if H<5:break
		return I,C
	Y,Z=X();return C,Y,Z
def relocate_operator(partial,steps=_A,mode=_H,uplift=1,seed=_A,verbose=_B):
	E=partial;B=steps;C=E.problem.K
	if C<2:return E.copy(),[_B]*C,0
	if B==_A:B=10**9
	M=random.Random(seed);A=E.copy();H=[_B]*C;D=0
	while D<B:
		I=list(enumerate(A.route_costs));F=max(I,key=lambda x:x[1])[0];N=[A for(A,B)in sorted(I,key=lambda x:x[1])]
		if len(A.routes[F])<5:break
		J=_B
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
def _default_vfunc(partial,sample_size=6,w_std=.15,seed=_A):'Default value function: negative max route cost.';return-balanced_scorer(partial,sample_size=sample_size,w_std=w_std,seed=seed)
def _default_selpolicy(actions,seed=_A,t=.1):
	'Default selection policy: choose action with minimal incremental cost.';A=actions;B=random.Random(seed)
	if not A:return
	C=softmax_weighter([action_weight(A)for A in A],t=t);D=sample_from_weight(B,C);return A[D]
def _default_simpolicy(partial,seed=_A):'Default simulation policy: greedy balanced solver.';B=partial;A=seed;C,D=greedy_solver(B.problem,partial=B);assert C is not _A,'Greedy solver failed in simulation policy.';A=107*A+108 if A is not _A else _A;return PartialSolution.from_solution(C)
def _default_defpolicy(partial,verbose=_B,seed=_A):'Default defense policy: beam search solver.';A=partial;B,C=iterative_greedy_solver(A.problem,partial=A,iterations=2000,time_limit=2e1,verbose=verbose);return B
@dataclass
class RewardFunction:
	visits:int=0;min_value:float=float('inf');max_value:float=float(_Q)
	def update(A,value):
		B=value
		if not math.isfinite(B):return
		A.visits+=1;A.min_value=min(A.min_value,B);A.max_value=max(A.max_value,B)
	def reward_from_value(A,value,reward_pow=_J):
		D=value;B=reward_pow
		if not math.isfinite(D):return .0
		if A.visits==0:return .5**B
		if A.max_value==A.min_value:return .5**B
		E=A.max_value-A.min_value;C=(D-A.min_value)/E;C=C**B;return max(.0,min(_J,C))
@dataclass
class MCTSNode:
	partial:PartialSolution;parent:Optional[_Y]=_A;action:Optional[Action]=_A;width:Optional[int]=_A;children:List[_Y]=field(default_factory=list);visits:int=0;total_cost:int=0;total_reward:float=.0;untried_actions:List[Action]=field(default_factory=list)
	def __post_init__(A):A.untried_actions=enumerate_actions_greedily(A.partial,A.width)
	@property
	def is_terminal(self):return self.partial.is_completed()
	@property
	def average_reward(self):
		A=self
		if A.visits==0:return .0
		return A.total_reward/A.visits
	@property
	def average_cost(self):
		A=self
		if A.visits==0:return .0
		return A.total_cost/A.visits
	def uct_score(A,uct_c):
		if A.visits==0:return float('inf')
		B=A.average_reward;C=A.parent.visits if A.parent else A.visits;D=uct_c*math.sqrt(math.log(C+1)/A.visits);return B+D
	def update(A,cost,reward):A.visits+=1;A.total_cost+=cost;A.total_reward+=reward
def _select(root,exploration):
	B=[root];A=root
	while _C:
		if A.untried_actions:return B
		if not A.children:return B
		A=max(A.children,key=lambda child:child.uct_score(exploration));B.append(A)
def _expand(node,selection_policy,width):
	A=node
	if not A.untried_actions:return
	B=selection_policy(A.untried_actions)
	if B is _A:return
	try:A.untried_actions.remove(B)
	except ValueError:pass
	C=A.partial.copy();apply_general_action(C,B);D=MCTSNode(C,parent=A,action=B,width=width);A.children.append(D);return D
def _backpropagate(path,cost,reward):
	for A in reversed(path):A.update(cost,reward)
def _gather_leaves(node,value_function,limit=_A):
	A=limit
	if A is _A:A=10**9
	assert A is not _A and A>0,'Limit must be positive';B=[];G=count()
	def D(current):
		C=current
		if not C.children:
			E=value_function(C.partial);F=E,next(G),C
			if len(B)<A:heapq.heappush(B,F)
			elif E>B[0][0]:heapq.heapreplace(B,F)
			return
		for H in C.children:D(H)
	D(node);C=sorted(B,key=lambda item:item[0],reverse=_C);return[A[2]for A in C]
def _run_mcts(problem,partial,width,uct_c,cutoff_depth,cutoff_depth_inc,cutoff_iter,reward_pow,value_function,selection_policy,simulation_policy,defense_policy,seed,time_limit,verbose):
	e='best_rollout_cost';Z=time_limit;Y=cutoff_depth;X=problem;O=partial;J=width;G=seed;C=verbose;P=time.time();f=P+Z;E=RewardFunction()
	if G is not _A:random.seed(G)
	if O is _A:O=PartialSolution(problem=X,routes=[])
	g=O or PartialSolution(problem=X,routes=[]);I=MCTSNode(g,width=J);A=0;B=_A;K=_A;D=10**18;L=0;a=Y;b=0;c=_T
	while _C:
		if time.time()>=f:
			c=_X
			if C:print(f"[MCTS] Reached time limit: {Z:.2f}s")
			break
		A+=1;Q=_select(I,uct_c);H=Q[-1];F=H.partial.num_actions
		if F>L:
			L=F
			if F>0 and F==a:
				b+=1;a+=Y+cutoff_depth_inc*b;I=MCTSNode(H.partial,width=J);E=RewardFunction()
				if C:print(f"[MCTS] Cutoff at iter {A}, abs_depth {F}, new root set.")
				continue
		if A>0 and A%cutoff_iter==0:
			I=MCTSNode(H.partial,width=J);E=RewardFunction()
			if C:print(f"[MCTS] Cutoff at iteration {A}, depth {F}, new root set.")
			continue
		if not H.is_terminal:
			R=_expand(H,selection_policy,J)
			if R is not _A:Q.append(R);M=R
			else:M=H
		else:break
		N=simulation_policy(M.partial.copy(),seed=12*G if G is not _A else _A)
		if N is _A or not N.is_completed():
			if C:print(f"[MCTS] Rollout failed or incomplete at iteration {A}.")
			E.update(float(_Q));continue
		S=N.to_solution();assert S is not _A,'Conversion from rollout to solution failed.';T=S.max_cost
		if T<=D:
			if B is _A or M.partial.num_actions>B.num_actions:B=M.partial.copy();D=T;K=S
		d=value_function(N);E.update(d);h=E.reward_from_value(d,reward_pow=reward_pow);_backpropagate(Q,T,h)
		if C:
			i=10**(len(str(A))-1)
			if A%i==0 or A%1000==0:j=time.time()-P;print(f"[MCTS] [Iteration {A}] Cost: {D:.3f}, Value range: {E.min_value:.3f} - {E.max_value:.3f}, Depth: {F}, Max depth: {L}, Time: {j:.2f}s.")
	U={_L:A,_K:time.time()-P,e:D,_O:c}
	if C:print(f"[MCTS] Iterations count: {A}, Max absolute depth reached: {L}, Time={U[_K]:.3f}s.");print(f"[MCTS] Best leaf depth: {B.num_actions if B else'N/A'} with rollout cost: {D:.3f}.")
	if B is _A:B=I.partial
	if B is not _A and B.is_pending():
		if C:print(f"[MCTS] Applying defense policy on best leaf...")
		V=defense_policy(B,verbose=C,seed=24*G if G is not _A else _A)
		if V is not _A:
			W=V.max_cost
			if K is _A or W<D:
				if C:print(f"[MCTS] Defense policy improved solution: {D:.3f} -> {W:.3f}")
				K=V;D=W;U[e]=D
	return I,B,K,U
def mcts_enumerator(problem,partial=_A,n_return=5,width=3,uct_c=.58,cutoff_depth=9,cutoff_depth_inc=4,cutoff_iter=11300,reward_pow=1.69,value_function=_default_vfunc,selection_policy=_default_selpolicy,simulation_policy=_default_simpolicy,defense_policy=_default_defpolicy,seed=_A,time_limit=3e1,verbose=_B):A=value_function;B,C,C,D=_run_mcts(problem,partial,width=width,uct_c=uct_c,cutoff_depth=cutoff_depth,cutoff_depth_inc=cutoff_depth_inc,cutoff_iter=cutoff_iter,reward_pow=reward_pow,value_function=A,selection_policy=selection_policy,simulation_policy=simulation_policy,defense_policy=defense_policy,time_limit=time_limit,seed=seed,verbose=verbose);E=_gather_leaves(B,value_function=A,limit=max(1,n_return));F=[A.partial.copy()for A in E];return F,D
def mcts_solver(problem,partial=_A,width=3,uct_c=.58,cutoff_depth=9,cutoff_depth_inc=4,cutoff_iter=11300,reward_pow=1.69,value_function=_default_vfunc,selection_policy=_default_selpolicy,simulation_policy=_default_simpolicy,defense_policy=_default_defpolicy,seed=_A,time_limit=3e1,verbose=_B):
	F='------------------------------';D=seed;C=verbose;G=time.time();E,K,B,A=_run_mcts(problem=problem,partial=partial,value_function=value_function,selection_policy=selection_policy,simulation_policy=simulation_policy,defense_policy=defense_policy,width=width,uct_c=uct_c,cutoff_depth=cutoff_depth,cutoff_depth_inc=cutoff_depth_inc,cutoff_iter=cutoff_iter,reward_pow=reward_pow,seed=D,time_limit=time_limit,verbose=C);assert B
	if C:print(f"[MCTS] Applying relocate operator to final solution...")
	H=PartialSolution.from_solution(B);I,E,E=relocate_operator(H,mode=_H,seed=_A if D is _A else 4*D+123);B=I.to_solution();assert B;J=B.max_cost
	if C:print(f"[MCTS] After relocate, final solution cost: {J}")
	A['used_best_rollout']=_C;A[_L]=A.get(_L,0);A[_K]=time.time()-G
	if C:
		if B is not _A:print();print(f"[MCTS] Final solution cost: {B.max_cost:.3f} after {A[_L]} iterations in {A[_K]:.2f}s.");print(F);print()
		else:print();print(f"[MCTS] No solution found after {A[_L]} iterations in {A[_K]:.2f}s.");print(F);print()
	return B,A
def read_instance():
	A,B,D=map(int,sys.stdin.readline().strip().split());E=list(map(int,sys.stdin.readline().split()));F=list(map(int,sys.stdin.readline().split()));C=[[0]*(2*A+2*B+1)for C in range(2*A+2*B+1)]
	for G in range(2*A+2*B+1):H=sys.stdin.readline().strip();C[G]=list(map(int,H.split()))
	return ShareARideProblem(A,B,D,E,F,C)
def main(verbose=_B):
	G=read_instance();A=G.num_nodes;A=G.num_nodes
	if A<=100:B,C,D,E,F=9,3,9000,6,.6
	elif A<=250:B,C,D,E,F=8,2,4000,4,.5
	elif A<=500:B,C,D,E,F=7,1,1400,3,.3
	elif A<=1000:B,C,D,E,F=6,0,600,2,.1
	else:B,C,D,E,F=5,0,250,2,.05
	H,I=mcts_solver(G,width=E,uct_c=F,cutoff_depth=B,cutoff_depth_inc=C,cutoff_iter=D,seed=42,time_limit=25e1,verbose=verbose);assert H;H.stdin_print()
if __name__=='__main__':main(verbose=_B)