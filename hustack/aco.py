_V='average_cost'
_U='avg_cost'
_T='done'
_S='return'
_R='elitists_count'
_Q='actions_done'
_P='dropL'
_O='pickL'
_N='best_cost'
_M='actions'
_L='iterations'
_K='serveP'
_J='status'
_I='time'
_H=1.
_G='parcels'
_F='load'
_E='pos'
_D='ended'
_C=True
_B=None
_A=False
import sys,math,random,time,bisect
from dataclasses import dataclass
from typing import Any,Callable,Dict,List,Optional,Tuple,Union,Sequence,Iterator,Set
Request=Tuple[int,int,str]
ActionNode=Tuple[int,int]
ValueFunction=Callable
FinalizePolicy=Callable
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
			Q=E[-1];R=M>1 and E[-1]==0;S={_E:Q,_G:F,_F:H,_M:D,_D:R};L.append(S)
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
			if O[_G]!=G:return _A
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
		return sorted(tuple(A)for A in A.routes)==sorted(tuple(A)for A in B.routes)
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
		A=E.problem;G=F[_E];C=[]
		for I in E.remaining_pass_serve:J,K=A.pserve(I);D=A.D[G][J]+A.D[J][K];C.append((_K,I,D))
		for B in E.remaining_parc_pick:
			L=A.q[B-1]
			if F[_F]+L<=A.Q[H]:D=A.D[G][A.lpick(B)];C.append((_O,B,D))
		for B in F[_G]:D=A.D[G][A.ldrop(B)];C.append((_P,B,D))
		C.sort(key=lambda x:x[2]);return C
	def check_expand(A,route_idx,kind,actid):
		E=route_idx;C=actid;B=kind;D=A.states[E];F=A.problem
		if D[_D]:return _A
		if B==_K:return C in A.remaining_pass_serve
		if B==_O:return C in A.remaining_parc_pick and D[_F]+F.q[C-1]<=F.Q[E]
		if B==_P:return C in D[_G]
		raise ValueError(f"Unknown action kind: {B}")
	def check_return(B,route_idx):A=B.states[route_idx];return not(A[_D]or A[_G])
	def apply_extend(A,route_idx,kind,actid,inc):
		G=kind;C=actid;B=route_idx;I=A.routes[B];D=A.states[B];E=A.problem
		if D[_D]:raise ValueError(f"Cannot apply action on ended route {B}.")
		if G==_K:K,J=E.pserve(C);I.append(K);I.append(J);A.node_assignment[K]=B;A.node_assignment[J]=B;A.remaining_pass_serve.discard(C);D[_E]=J;D[_M]+=1;A.route_costs[B]+=inc;A.max_cost=max(A.max_cost,A.route_costs[B]);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions+=1;return
		elif G==_O:
			F=E.q[C-1]
			if D[_F]+F>E.Q[B]:raise ValueError(f"Taxi {B} capacity exceeded for parcel {C}.")
			H=E.lpick(C);D[_F]+=F;D[_G].add(C);A.remaining_parc_pick.discard(C);A.remaining_parc_drop.add(C)
		elif G==_P:
			F=E.q[C-1]
			if D[_F]-F<0:raise ValueError(f"Taxi {B} load cannot be negative after dropping parcel {C}.")
			H=E.ldrop(C);D[_F]-=F;D[_G].discard(C);A.remaining_parc_drop.discard(C)
		else:raise ValueError(f"Unknown action kind: {G}")
		D[_E]=H;D[_M]+=1;I.append(H);A.node_assignment[H]=B;A.route_costs[B]+=inc;A.max_cost=max(A.max_cost,A.route_costs[B]);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions+=1
	def apply_return(A,t_idx):
		C=t_idx;D=A.routes[C];B=A.states[C]
		if B[_D]:return
		if B[_E]==0 and B[_M]>0:B[_D]=_C;return
		if B[_G]:raise ValueError(f"Taxi {C} must drop all loads before returning to depot.")
		E=A.problem.D[B[_E]][0];D.append(0);B[_E]=0;B[_M]+=1;B[_D]=_C;A.route_costs[C]+=E;A.max_cost=max(A.max_cost,A.route_costs[C]);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions+=1
	def reverse_action(A,t_idx):
		G=t_idx;F=A.routes[G];C=A.states[G]
		if len(F)<=1:raise ValueError(f"No actions to reverse for taxi {G}.")
		B=A.problem;D=F[-1]
		if B.is_pdrop(D):
			J=F.pop();I=F.pop();L=B.rev_pdrop(J)
			if B.rev_ppick(I)!=L:raise ValueError('Inconsistent route state: pdrop not preceded by corresponding ppick.')
			H=F[-1];K=B.D[H][I]+B.D[I][J];C[_E]=H;C[_M]-=1;C[_D]=_A;A.remaining_pass_serve.add(L);A.node_assignment[J]=-1;A.node_assignment[I]=-1;A.route_costs[G]-=K;A.max_cost=max(A.route_costs);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions-=1;return
		D=F.pop();H=F[-1];K=B.D[H][D];C[_E]=H;C[_M]-=1;C[_D]=_A
		if B.is_lpick(D):E=B.rev_lpick(D);C[_F]-=B.q[E-1];C[_G].discard(E);A.remaining_parc_pick.add(E);A.remaining_parc_drop.discard(E)
		elif B.is_ldrop(D):E=B.rev_ldrop(D);C[_F]+=B.q[E-1];C[_G].add(E);A.remaining_parc_pick.discard(E);A.remaining_parc_drop.add(E)
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
class PartialSolutionSwarm:
	def __init__(A,solutions):
		B=solutions
		if not B:raise ValueError('Solutions list cannot be empty.')
		A.problem=B[0].problem;A.num_partials=len(B);A.partial_lists=B;A.partial_num_actions=[A.num_actions for A in B];A.partial_costs=[A.max_cost for A in B];A.min_cost=min(A.partial_costs);A.max_cost=max(A.partial_costs);A.avg_cost=sum(A.max_cost for A in B)/len(B)
	def update(A):A.partial_num_actions=[A.num_actions for A in A.partial_lists];A.partial_costs=[A.max_cost for A in A.partial_lists];A.min_cost=min(A.partial_costs);A.max_cost=max(A.partial_costs);A.avg_cost=sum(A.max_cost for A in A.partial_lists)/len(A.partial_lists)
	def opt(B):
		B.update();C=10**18;D=_B
		for E in B.partial_lists:
			if E.is_completed():
				A=E.to_solution()
				if A and A.max_cost<C:C=A.max_cost;D=A
		return D
	def stats(A):A.update();return{'num_partials':A.num_partials,'min_cost':A.min_cost,'max_cost':A.max_cost,_U:A.avg_cost}
	def copy(A):B=[A.copy()for A in A.partial_lists];return PartialSolutionSwarm(solutions=B)
def weighted(kind,inc,pweight=.7):
	if kind==_K:return pweight*inc
	return inc+1
def action_weight(action,pweight=.7):A=action;return weighted(A[1],A[3],pweight)
def softmax_weighter(incs,t):
	A=incs;B,E=min(A),max(A);C=E-B
	if C<1e-06:return[_H]*len(A)
	D=[]
	for F in A:G=(F-B)/C;D.append((_H-G+.1)**(_H/t))
	return D
def balanced_scorer(partial,sample_size=8,w_std=.15,seed=_B):
	B=sample_size;A=partial;E=random.Random(seed);C=sorted(A.route_costs)
	if len(C)==1:return A.max_cost
	D=E.choices(C,k=B);F=sum(D)/B;G=sum((A-F)**2 for A in D)/B;H=G**.5;return A.max_cost+w_std*H
def check_general_action(partial,action):
	A=partial;B,C,D,E=action
	if C==_S:return A.check_return(B)
	return A.check_expand(B,C,D)
def apply_general_action(partial,action):
	A=partial;B,C,D,E=action
	if C==_S:A.apply_return(B)
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
			if A.check_return(C):Z=B.D[Y[_E]][0];S.append((C,_S,0,Z))
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
def repair_one_route(partial,route_idx,steps,T=_H,seed=_B,verbose=_A):
	B=route_idx;A=partial;E=random.Random(seed);C=0
	for M in range(steps):
		F=A.states[B]
		if F[_D]:break
		D=A.possible_expand(B)
		if not D:A.apply_return(B);C+=1;break
		G=[weighted(A,B)for(A,C,B)in D];H=softmax_weighter(G,T);I=sample_from_weight(E,H);J,K,L=D[I];A.apply_extend(B,J,K,L);C+=1
	return A,C
def repair_operator(partial,repair_proba=_B,steps=_B,T=_H,seed=_B,verbose=_A):
	C=steps;B=repair_proba;A=partial;E=random.Random(seed)
	if B is _B:B=_H
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
def destroy_operator(sol,destroy_proba,destroy_steps,seed=_B,t=_H,verbose=_A):
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
def greedy_solver(problem,partial=_B,verbose=_A):
	A=partial;C=time.time()
	if A is _B:A=PartialSolution(problem=problem)
	B=0;D=A.num_actions
	while A.is_pending():
		B+=1;E=enumerate_actions_greedily(A,1)
		if not E:return _B,{_L:B,_I:time.time()-C,_Q:A.num_actions-D,_J:'error'}
		F=E[0];apply_general_action(A,F)
	G=A.to_solution();H={_L:B,_I:time.time()-C,_Q:A.num_actions-D,_J:_T};return G,H
def iterative_greedy_solver(problem,partial=_B,iterations=10000,destroy_proba=.53,destroy_steps=13,destroy_t=1.3,rebuild_proba=.29,rebuild_steps=3,rebuild_t=1.2,time_limit=3e1,seed=_B,verbose=_A):
	F=seed;E=partial;D=problem;G=time.time();K=G+time_limit;T=random.Random(F)
	if E is _B:E=PartialSolution(problem=D,routes=[])
	B,U=greedy_solver(D,partial=E,verbose=verbose)
	if not B:return _B,{_I:time.time()-G,_J:'error'}
	H=B.max_cost;C=U[_Q];L=H;M=0;N=0;O=0;P=_T;I=0
	for V in range(1,iterations+1):
		if K and time.time()>=K:P='overtime';break
		Q=_B if F is _B else 2*F+98*V;W,X,R=destroy_operator(B,destroy_proba,destroy_steps,seed=Q,t=destroy_t);N+=R;C+=R;J=W
		for(Y,Z)in enumerate(X):
			if not Z:continue
			if T.random()>rebuild_proba:continue
			J,S=repair_one_route(J,route_idx=Y,steps=rebuild_steps,T=rebuild_t,seed=Q);O+=S;C+=S
		A,a=greedy_solver(D,partial=J,verbose=_A);L+=A.max_cost if A else 0;C+=a[_Q];I+=1 if A else 0
		if A and A.max_cost<H:B=A;H=A.max_cost;M+=1
	b=time.time()-G;c={_L:I,_Q:C,'improvements':M,'actions_destroyed':N,'actions_rebuilt':O,_V:L/(I+1),_I:b,_J:P};return B,c
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
	'\n    Compute the decrement in max_cost of relocating a full request (pickup and drop)\n    from one route to another at specified insertion indices, as Cost_before - Cost_after.\n    This assumes the pickup/drop indices are correct with some defensive assertions.\n\n    Parameters:\n    - partial: PartialSolution object representing the current solution.\n    - from_route_idx: Index of the route from which the request is relocated.\n    - to_route_idx: Index of the route to which the request is relocated.\n    - pfidx: Index of the pickup node in the from_route.\n    - qfidx: Index of the drop node in the from_route.\n    - ptidx: Index in the to_route where the pickup node will be inserted.\n    - qtidx: Index in the to_route where the drop node will be inserted.\n\n    Returns: A tuple containing:\n    - from_route_next_cost: Cost of the from_route after relocation.\n    - to_route_next_cost: Cost of the to_route after relocation.\n    - cost_decrement: Decrement in max_cost due to the relocation.\n    ';R=qtidx;Q=ptidx;P=rtidx;O=rfidx;I=qfidx;H=pfidx;B=partial;A=B.problem.D;Y=B.max_cost;E=B.routes[O];J=B.routes[P];Z=B.route_costs[O];a=B.route_costs[P];C=E[H];D=E[I];K=E[H-1];S=E[H+1];T=E[I-1];L=E[I+1];F=0
	if H+1==I:F-=A[K][C]+A[C][D]+A[D][L];F+=A[K][L]
	else:F-=A[K][C]+A[C][S]+A[T][D]+A[D][L];F+=A[K][S]+A[T][L]
	U=Z+F;M=J[Q-1];V=J[Q];W=J[R-2];N=J[R-1];G=0
	if R==Q+1:G-=A[M][N];G+=A[M][C]+A[C][D]+A[D][N]
	else:G-=A[M][V]+A[W][N];G+=A[M][C]+A[C][V]+A[W][D]+A[D][N]
	X=a+G;b=[B.route_costs[A]for A in range(B.problem.K)if A!=O and A!=P];c=max(U,X,*b);return U,X,Y-c
def relocate_from_to(partial,route_from_idx,route_to_idx,steps,mode,uplift=1,seed=_B,verbose=_A):
	'\n    Attempt to relocate requests from one predefined vehicle route to another that\n    improves the solution as a helper to the main relocate_operator.\n\n    Parameters:\n    - partial: PartialSolution object representing the current solution.\n    - from_route_idx: Index of the route from which to relocate requests.\n    - to_route_idx: Index of the route to which to relocate requests.\n    - steps: Number of steps to consider\n    - mode: Mode of operation\n    - uplift: Integer controlling the extent of improvement required.\n    - seed: Random seed for stochastic modes.\n    - verbose: If True, print detailed logs.\n    ';N='serveL';L=mode;K=partial;F=route_from_idx;E=route_to_idx;O=random.Random(seed);A=K.problem;C=K.copy();B=C.routes[F];D=C.routes[E];H=len(B);G=len(D)
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
		"\n        Ensure that pickup and drop indices are consecutive for 'serveP' requests.\n        ";D,E,A,B,C=req
		if C==N:return _C
		return B==A+1
	def S(req):
		'\n        Ensure load stays within [0,cap] after relocating a full request\n        (pfidx, qfidx) to (ptidx, qtidx) for both routes.\n        ';D,E,F,G,K=req
		if K==_K:return _C
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
		'\n        Check feasibility of relocating the request defined by (pfidx, qfidx)\n        from route_from to route_to at insertion indices (ptidx, qtidx).\n        \n        Returns a tuple of:\n            - feasibility (bool)\n            - after_cost_a (int): cost of from_route after relocation\n            - after_cost_b (int): cost of to_route after relocation\n            - dec (int): total cost decrement if relocation is performed\n        ';A=req
		if not R(A):return
		if not S(A):return
		B=cost_decrement_relocate(C,F,E,A[0],A[1],A[2],A[3]);return B
	def U():
		'\n        Find candidate relocation requests according to the specified mode.\n        \n        Yields tuples of the form (Request, CostChange).\n        ';M={B:A for(A,B)in enumerate(B)};H=[]
		for(E,I)in enumerate(B[1:],start=1):
			if A.is_ppick(I):C=E+1;H.append((E,C,_K))
			elif A.is_lpick(I):
				O=A.rev_lpick(I);P=A.ldrop(O);C=M.get(P)
				if C is not _B and C>E:H.append((E,C,N))
		Q=[(B,B+1)for B in range(1,G)if not A.is_ppick(D[B-1])];R=[(B,C)for B in range(1,G)if not A.is_ppick(D[B-1])for C in range(B+1,G+1)if not A.is_ppick(D[C-2])]
		for(E,C,K)in H:
			S=Q if K==_K else R
			for(U,V)in S:
				J=E,C,U,V,K;F=T(J)
				if F is _B:continue
				X,Y,W=F
				if W<uplift:continue
				if L=='first':yield(J,F);return
				else:yield(J,F)
	def V():
		'\n        Select a candidate relocation based on the specified mode.\n        ';A=list(U())
		if not A:return
		if L=='stochastic':return O.choice(A)
		elif L=='best':return max(A,key=lambda x:x[1][2])
		else:return A[0]
	def W(action):'\n        Apply relocation to routes and update costs / max cost for the\n        current partial solution object.\n        ';nonlocal B,D,C;(A,G,J,K,O),(L,M,N)=action;H=B[A];I=B[G];del B[G];del B[A];D.insert(J,H);D.insert(K,I);C.routes[F]=B;C.routes[E]=D;C.route_costs[F]=L;C.route_costs[E]=M;C.max_cost-=N;C.node_assignment[H]=E;C.node_assignment[I]=E
	def X(action):
		'\n        Incrementally update passenger & load delta managers after a relocation\n        using MinMaxPfsumArray insert/delete operations (avoid full rebuild).\n\n        action: (p_from, q_from, p_to, q_to, new_cost_from, new_cost_to, dec)\n        Indices p_from,q_from,p_to,q_to refer to ORIGINAL pre-mutation routes.\n        Relocation sequence applied earlier in update_partial_solution:\n            1. Remove q_from, then p_from from donor route_from.\n            2. Insert pickup at p_to in route_to.\n            3. Insert drop at drop_insert_index = (q_to if q_to was final depot else q_to+1).\n        Here we commit those operations on the delta managers.\n        ';nonlocal I,J,B,D;(C,E,G,H,M),N=action;K=B[C];L=B[E]
		def F(nodeid):
			B=nodeid
			if A.is_lpick(B):C=A.rev_lpick(B);return A.q[C-1]
			elif A.is_ldrop(B):C=A.rev_ldrop(B);return-A.q[C-1]
			else:return 0
		I.delete(E);I.delete(C);J.insert(G,F(K));J.insert(H,F(L))
	def Y():
		'\n        Perform relocation steps until no further improvement is possible\n        or the specified number of steps is reached.\n\n        Returns a tuple of:\n        - modified_routes: List of booleans indicating which routes were modified.\n        - reloc_done: Number of relocations performed.\n        ';nonlocal H,G,B,D;C=0;I=[_A]*A.K
		while C<steps:
			J=V()
			if J is _B:break
			X(J);W(J);C+=1;I[F]=_C;I[E]=_C;H-=2;G+=2
			if H<5:break
		return I,C
	Z,a=Y();return C,Z,a
def relocate_operator(partial,steps=_B,mode='first',uplift=1,seed=_B,verbose=_A):
	"\n    Attempt to relocate the request from a vehicle route to another that\n    improves the solution to different extents controlled by ``uplift``.\n    Perform up to ``steps`` relocations based on the specified ``mode``.\n    Use in post-processing or local search.\n\n    The procedure attempts to relocate requests from the highest-cost route\n    to the 1/3 lower-cost routes (traverse from the lowest to the highest) by\n    iterating over all insertion pairs of the receiver routes, in ascending \n    order of their in-out contribution to the route cost.\n    If not successful, it moves to the next donor route.\n\n    Parameters:\n    - partial: PartialSolution object representing the current solution.\n    - steps: Number of relocation steps that the operation should perform.\n    - mode: Mode of operation, can be 'best', 'first', or 'stochastic'.\n    - uplift: Integer controlling the extent of improvement required.\n    - seed: Random seed for stochastic modes.\n    - verbose: If True, print detailed logs.\n\n    Returns: a tuple signature containing:\n    - A new PartialSolution object with the specified requests relocated.\n    - A list of booleans indicating which routes were modified.\n    - An integer count of the number of relocations performed.\n    ";E=partial;B=steps;C=E.problem.K
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
def _default_value_function(partial,perturbed_samples=6,seed=_B):A=partial;C,B=C,B=iterative_greedy_solver(A.problem,A,iterations=perturbed_samples,time_limit=.1,seed=seed);return B[_V]
def _default_finalize_policy(partial,seed=_B):
	A=partial;B,E=iterative_greedy_solver(A.problem,A,iterations=3000,time_limit=3.,seed=seed,verbose=_A)
	if not B:return
	C=PartialSolution.from_solution(B);D,F,G=relocate_operator(partial=C,seed=seed,verbose=_A);return D.to_solution()
@dataclass
class SolutionTracker:
	best_solution:Optional[Solution]=_B;best_cost:int=10**18;worst_cost:int=-1;total_cost:int=0;count:int=0
	def update(B,source):
		A=source
		if isinstance(A,Solution):B._update_from_solution(A)
		elif isinstance(A,PartialSolutionSwarm):B._update_from_swarm(A)
		elif isinstance(A,list):B._update_from_list(A)
	def _update_from_solution(A,solution):
		'Update metrics with a single solution.';C=solution;B=C.max_cost;A.count+=1;A.total_cost+=B;A.worst_cost=max(A.worst_cost,B)
		if B<A.best_cost:A.best_cost=B;A.best_solution=C
	def _update_from_swarm(C,swarm):
		'Update metrics with a swarm of partial solutions.'
		for A in swarm.partial_lists:
			if not A.is_completed():continue
			B=A.to_solution()
			if not B:continue
			C._update_from_solution(B)
	def _update_from_list(B,solutions):
		'Update metrics with a list of solutions.'
		for A in solutions:
			if A is _B:continue
			B._update_from_solution(A)
	def stats(A):'Return statistics of the population.';B=A.total_cost/A.count if A.count>0 else .0;return{_N:A.best_cost,'worst_cost':A.worst_cost,_U:B,'count':A.count}
	def opt(A):'Return the best solution found so far.';return A.best_solution
class PheromoneMatrix:
	def __init__(A,problem,sigma,rho,init_cost):B=sigma;A.size=problem.num_nodes;A.sigma=B;A.rho=rho;A.tau_0=_H/(rho*init_cost);A.tau_max=2*A.tau_0;A.tau_min=A.tau_0/1e1;A.tau=[[A.tau_0 for B in range(A.size)]for B in range(A.size)];assert .0<A.rho<_H,'Evaporation rate rho must be in (0,1).';assert B>=1,'Number of elitists sigma must be at least 1.'
	def get(A,prev,curr):'Get pheromone level on transition from prev action to curr action.';return A.tau[prev[1]][curr[0]]
	def update(A,swarm,opt):
		D=opt
		def F(partial):
			"Extract all (prev_out, curr_in) edges from a partial solution's routes.";A=partial;B=[]
			for(D,G)in enumerate(A.routes):
				C=A.enumerate_action_nodes(D)
				for(E,F)in zip(C[:-1],C[1:]):B.append((E[1],F[0]))
			return B
		G=sorted(((A.max_cost,A)for A in swarm.partial_lists),key=lambda x:x[0])[:A.sigma];A.tau=[[max(A.tau_min,A.rho*B)for B in B]for B in A.tau];E=set()
		for(H,(I,J))in enumerate(G):
			K=(A.sigma-H)/I
			for(B,C)in F(J):
				E.add((B,C))
				if 0<=B<A.size and 0<=C<A.size:A.tau[B][C]+=K
		if D is not _B and D.max_cost>0:
			L=A.sigma/D.max_cost;M=PartialSolution.from_solution(D)
			for(B,C)in F(M):
				E.add((B,C))
				if 0<=B<A.size and 0<=C<A.size:A.tau[B][C]+=L
		for(B,C)in E:A.tau[B][C]=min(A.tau_max,A.tau[B][C])
class DesirabilityMatrix:
	def __init__(A,problem,phi,chi,gamma,kappa):
		F=problem;A.size=F.num_nodes;A.problem=F;A.phi=phi;A.chi=chi;A.gamma=gamma;A.kappa=kappa;A.saving_matrix=[];B=A.problem.D
		for C in range(A.size):
			D=[]
			for E in range(A.size):
				if C==E:D.append(0)
				else:G=max(B[0][C]+B[E][0]-B[C][E],0);H=(1+G)**A.phi;D.append(H)
			A.saving_matrix.append(D)
	def get(A,prev,curr,partial,action):
		D,B,E,G=action;F=A.problem.Q[D];H=partial.states[D];C=H[_F]
		if B==_O:C+=A.problem.q[E-1]
		if B==_P:C-=A.problem.q[E-1]
		I=(1+weighted(B,G))**A.chi;J=A.saving_matrix[prev[1]][curr[0]];K=2-int(B=='serveP');L=(1+A.gamma*(F-C)/F)*A.kappa;return J/I*K*L
class NearestExpansionCache:
	def __init__(C,problem,n_nearest=3):
		A=problem;C.nearest_actions=[]
		for B in range(A.num_nodes):
			if B==0:D=[[0]for A in range(A.K)]
			elif A.is_ppick(B):C.nearest_actions.append([]);continue
			elif A.is_pdrop(B):G=A.rev_pdrop(B);F=A.pserve(G)[0];D=[[0,F,B]]+[[0]for A in range(A.K-1)]
			elif A.is_lpick(B):D=[[0,B]]+[[0]for A in range(A.K-1)]
			elif A.is_ldrop(B):H=A.rev_ldrop(B);F=A.lpick(H);D=[[0,F,B]]+[[0]for A in range(A.K-1)]
			else:print(f"[ACO] [Error] Cache error: Unknown node type for node {B}.");C.nearest_actions.append([]);continue
			I=PartialSolution(A,routes=D);E=I.possible_expand(0);E.sort(key=lambda item:weighted(item[0],item[2]));E=E[:n_nearest];C.nearest_actions.append(E)
	def query(K,partial,n_queried):
		A=partial
		if A.num_actions<A.problem.num_expansions:return[]
		L=A.max_cost;B=[];C=[]
		for(D,G)in enumerate(A.states):
			if G[_D]:continue
			M=G[_E];N=K.nearest_actions[M]
			for O in N:
				E,H,F=O
				if not A.check_expand(D,E,H):continue
				I=D,E,H,F;J=weighted(E,F)
				if A.route_costs[D]+F<=L:B.append((J,I))
				else:C.append((J,I))
		B.sort(key=lambda x:x[0]);C.sort(key=lambda x:x[0]);P=[A for(B,A)in B+C];return P[:n_queried]
class Ant:
	class ProbaExpandSampler:
		partial:PartialSolution;cache:'NearestExpansionCache';alpha:float;beta:float;omega:float;q_prob:float;width:int
		def __init__(A,partial,cache,alpha,beta,omega,q_prob,width):A.partial=partial;A.cache=cache;A.alpha=alpha;A.beta=beta;A.omega=omega;A.q_prob=q_prob;A.width=width
		def _get_action_node(E,action):
			'Get ActionNode for a given action.';F,B,C,G=action;D=E.partial.problem
			if B==_K:return D.pserve(C)
			elif B==_O:A=D.lpick(C);return A,A
			elif B==_P:A=D.ldrop(C);return A,A
			else:return 0,0
		def _collect_actions(F):
			B=F.partial;C=F.width
			if B.num_actions>=B.problem.num_expansions:A=enumerate_actions_greedily(B,C);D=action_weight(A[0]);return[(D/action_weight(A),A)for A in A]
			A=F.cache.query(B,C)
			if len(A)<C:A=enumerate_actions_greedily(B,C)[:C]
			G=[];H=_A;I=.0;J=A[0];D=action_weight(J);G.append((D,J))
			for(L,E)in zip(A[:-1],A[1:]):
				M=L[3];N=E[3]
				if not H and N>M:H=_C;O=action_weight(E);I=O+_H
				K=action_weight(E)
				if H:K+=I
				G.append((K,E))
			P=[(D/A,B)for(A,B)in G];return P
		def _compute_log_proba(A,tau,eta,fit,action):B=action;G=B[0];H=A.partial.states[G];I=H[_E];E=10**18,I;F=A._get_action_node(B);C=tau.get(E,F);D=eta.get(E,F,A.partial,B);C=max(C,1e-300);D=max(D,1e-300);J=+A.alpha*math.log(C)+A.beta*math.log(D)+A.omega*math.log(fit);return J
		def sample_action(A,tau,eta,rng):
			B=A._collect_actions()
			if not B:return
			C=[]
			for(G,H)in B:I=A._compute_log_proba(tau,eta,G,H);C.append(I)
			J=max(C);F=[math.exp(A-J)for A in C];K=sum(F);D=[A/K for A in F];E:0
			if rng.random()<A.q_prob:E=D.index(max(D))
			else:E=sample_from_weight(rng,D)
			return B[E][1]
	def __init__(A,partial,cache,tau,eta,alpha,beta,omega,q_prob,width,rng):'Initialize ant with parameters and partial solution.';G=width;F=q_prob;E=omega;D=alpha;C=cache;B=partial;A.problem=B.problem;A.partial=B;A.cache=C;A.tau=tau;A.eta=eta;A.alpha=D;A.beta=beta;A.omega=E;A.q_prob=F;A.width=G;A.rng=rng;A.sampler=Ant.ProbaExpandSampler(partial=A.partial,cache=C,alpha=D,beta=beta,omega=E,q_prob=F,width=G)
	def expand(A):
		if A.partial.is_completed():return _A
		B=A.sampler.sample_action(A.tau,A.eta,A.rng)
		if not B:return _A
		apply_general_action(A.partial,B);return _C
class AntPopulation:
	def __init__(A,swarm,cache,tau,eta,lfunc,alpha,beta,omega,q_prob,width,depth,time_limit,seed,verbose):
		D=cache;C=swarm;B=seed;A.swarm=C.copy();A.completed=[A.is_completed()for A in A.swarm.partial_lists];A.cache=D;A.tau=tau;A.eta=eta;A.lfunc=lfunc;A.depth=depth;A.time_limit=time_limit;A.seed=B;A.verbose=verbose;A.ants=[]
		for(E,F)in enumerate(A.swarm.partial_lists):G=Ant(partial=F,cache=D,tau=A.tau,eta=A.eta,alpha=alpha,beta=beta,omega=omega,q_prob=q_prob,width=width,rng=random.Random(hash(B+100*E)if B else _B));A.ants.append(G)
		A.num_ants=len(A.ants);A.max_actions=C.problem.num_actions;A.start_time=time.time();A.end_time=A.start_time+A.time_limit;A.tle=lambda:time.time()>A.end_time
	def expand(A):
		B=_A
		for(C,D)in enumerate(A.ants):
			if A.completed[C]:continue
			if D.expand():B=_C
			elif A.verbose:print(f"[ACO] [Depth {D.partial.num_actions}] [Warning] Ant {C+1} cannot expand, further diagnosis needed.");raise RuntimeError('Ant expansion failure.')
		return B
	def update(A):A.lfunc.update(source=A.swarm);A.tau.update(swarm=A.swarm,opt=A.lfunc.opt())
	def run(A):
		G='N/A';H=[A.max_actions//5,A.max_actions//2,A.max_actions*9//10]
		for B in range(A.depth):
			if A.tle():
				if A.verbose:print('[ACO] Time limit reached, skipping this iteration.')
				return A.swarm
			if A.verbose:
				if B in H or B%100==0:C=[A.max_cost for A in A.swarm.partial_lists];D=[A.num_actions for A in A.swarm.partial_lists];print(f"[ACO] [Iteration {B}] Partial cost range: {min(C):.3f} - {max(C):.3f}, Depth range: {min(D)} - {max(D)}, Time_elapsed={time.time()-A.start_time:.2f}s.")
			if not A.expand():
				if A.verbose:print('[ACO] All ants have completed their solutions.')
				break
			A.update()
		if A.verbose:I=sum(1 for A in A.swarm.partial_lists if A.is_completed());E=A.swarm.opt();F=A.lfunc.opt();print(f"[ACO] Finished all depth.\nComplete solutions found: {I}/{A.num_ants}.\nRun best cost: {E.max_cost if E else G}, Opt cost: {F.max_cost if F else G}.")
		return A.swarm
class SwarmTracker:
	def __init__(A,swarm,value_function,finalize_policy,seed=_B):C=value_function;B=swarm;A.seed=seed;A.frontier_swarm=[A.copy()for A in B.partial_lists];A.num_partials=B.num_partials;A.frontier_fitness=[C(A.copy())for A in A.frontier_swarm];(A.finals):0;A.is_finalized=_A;A.value_function=C;A.finalize_policy=finalize_policy
	def update(A,source):
		C=source;assert A.num_partials==C.num_partials
		for(B,D)in enumerate(C.partial_lists):
			E=A.value_function(D.copy(),seed=A.seed+10*B if A.seed else _B)
			if E<A.frontier_fitness[B]:A.frontier_swarm[B]=D.copy();A.frontier_fitness[B]=E
		return A.frontier_fitness
	def finalize(A,cutoff):
		C=cutoff;D=sorted(zip(A.frontier_swarm,A.frontier_fitness),key=lambda x:x[1]);F=D[:C]if C else D;B=[]
		for(G,(H,I))in enumerate(F):
			E=A.finalize_policy(H,seed=A.seed+20*G if A.seed else _B)
			if E:B.append(E)
		B.sort(key=lambda s:s.max_cost);B=B[:C]if C else B;A.is_finalized=_C;A.finals=B;return B
	def top(A,k,cutoff=_B):
		B=cutoff
		if B is _B:B=k
		if not A.is_finalized:A.finalize(B)
		return A.finals[:k]
	def opt(A,cutoff=_B):
		if not A.is_finalized:A.finalize(cutoff)
		return A.finals[0]
def _run_aco(problem,swarm,n_cutoff,iterations,depth,q_prob,alpha,beta,omega,phi,chi,gamma,kappa,sigma,rho,width,value_function,finalize_policy,seed,time_limit,verbose):
	H=time_limit;G=swarm;F=depth;E=seed;C=n_cutoff;B=problem;I=time.time()
	if F is _B:F=B.num_actions
	J,X=iterative_greedy_solver(problem=B,iterations=1000,time_limit=2.5,seed=10*E if E else _B,verbose=_A);O=J.max_cost;P=NearestExpansionCache(B,n_nearest=5);Q=PheromoneMatrix(B,sigma=sigma,rho=rho,init_cost=O);R=DesirabilityMatrix(B,phi,chi,gamma,kappa);D=SolutionTracker();D.update(J);A=SwarmTracker(swarm=G,value_function=value_function,finalize_policy=finalize_policy);K=0;S=_T
	for L in range(iterations):
		if time.time()-I>=.75*H:break
		T=AntPopulation(swarm=G,cache=P,tau=Q,eta=R,lfunc=D,alpha=alpha,beta=beta,omega=omega,q_prob=q_prob,width=width,depth=F,time_limit=H,seed=hash(E+10*L)if E else _B,verbose=verbose);M=T.run();A.update(M);D.update(M);K=L+1
	A.finalize(C)
	if D.best_solution:
		A.finals.append(D.best_solution);A.finals.sort(key=lambda s:s.max_cost)
		if C and len(A.finals)>C:A.finals=A.finals[:C]
	U=time.time()-I;N=A.opt(cutoff=C);V=N.max_cost if N else float('inf');W={_L:K,_N:V,_R:A.num_partials,_I:U,_J:S};return A,W
def aco_enumerator(problem,swarm=_B,n_partials=40,n_cutoff=10,n_return=5,iterations=10,depth=_B,q_prob=.54,alpha=1.14,beta=1.83,omega=4,phi=.36,chi=1.65,gamma=.43,kappa=2.03,sigma=11,rho=.81,width=5,value_function=_default_value_function,finalize_policy=_default_finalize_policy,seed=_B,time_limit=3e1,verbose=_A):
	'\n    Run ACO to enumerate the best k complete solutions.\n    \n    This function executes ACO with a SwarmTracker that tracks the best\n    variation of each partial across all depth. At the end, it finalizes\n    the top `n_cutoff` partials and returns the best `n_return` solutions.\n    \n    Args:\n        - problem: ShareARideProblem instance.\n        - swarm: Initial PartialSolutionSwarm (optional, created if None).\n\n        - n_partials: Number of ants (must match swarm size if provided).\n        - n_return: Number of best solutions to return.\n        - n_cutoff: Maximum partials to finalize (time-saving cutoff). Should be\n                >= n_return to avoid missing potentially better solutions.\n        - iterations: Number of ACO iterations to run.\n        - depth: Number of depth to run.\n        \n        - q_prob: Exploitation probability (0 = explore, 1 = exploit).\n        - alpha: Pheromone influence exponent.\n        - beta: Desirability influence exponent.\n        - phi: Savings influence exponent for desirability.\n        - chi: Distance influence exponent for desirability.\n        - gamma: Parcel influence factor for desirability.\n        - kappa: Parcel influence exponent for desirability.\n        - sigma: Number of elitists for pheromone update.\n        - rho: Evaporation rate for pheromone update.\n        - width: Maximum actions to consider per expansion.\n\n        - value_function: Function to evaluate potential of partial solutions.\n        - finalize_policy: Function to complete partial solutions.\n\n        - seed: Random seed for reproducibility.\n        - time_limit: Total time limit in seconds.\n        - verbose: If True, print detailed logs.\n    \n    Returns:\n        - solutions: List of top n_return solutions from finalized partials.\n        - info: Dictionary with run statistics.\n    ';D=n_cutoff;C=problem;B=swarm
	if B is _B:F=[PartialSolution(problem=C,routes=[])for A in range(n_partials)];B=PartialSolutionSwarm(solutions=F)
	G,A=_run_aco(problem=C,swarm=B,n_cutoff=D,iterations=iterations,depth=depth,q_prob=q_prob,alpha=alpha,beta=beta,omega=omega,phi=phi,chi=chi,gamma=gamma,kappa=kappa,sigma=sigma,rho=rho,width=width,value_function=value_function,finalize_policy=finalize_policy,seed=seed,time_limit=time_limit,verbose=verbose);E=G.top(k=n_return,cutoff=D);H={_L:A[_L],_I:A[_I],_N:A[_N],'solutions_found':len(E),_R:A[_R],_J:A[_J]};return E,H
def aco_solver(problem,swarm=_B,n_partials=40,n_cutoff=10,iterations=40,depth=_B,q_prob=.54,alpha=1.14,beta=1.83,omega=4,phi=.36,chi=1.65,gamma=.43,kappa=2.03,sigma=11,rho=.81,width=5,value_function=_default_value_function,finalize_policy=_default_finalize_policy,seed=_B,time_limit=3e1,verbose=_A):
	'\n    Run ACO solver and return the best complete solution.\n    \n    This function executes the ACO algorithm using SwarmTracker to track\n    the best partials across depth and returns the optimal solution.\n    \n    Args:\n        - problem: ShareARideProblem instance.\n        - swarm: Initial PartialSolutionSwarm (optional, created if None).\n\n        - n_partials: Number of ants (must match swarm size if provided).\n        - n_cutoff: Maximum partials to finalize (time-saving cutoff). Should be\n                > 1 to avoid missing potentially better solutions.\n        - iterations: Number of ACO iterations to run.\n        - depth: Number of depth to run.\n        \n        - q_prob: Exploitation probability (0 = explore, 1 = exploit).\n        - alpha: Pheromone influence exponent.\n        - beta: Desirability influence exponent.\n        - phi: Savings influence exponent for desirability.\n        - chi: Distance influence exponent for desirability.\n        - gamma: Parcel influence factor for desirability.\n        - kappa: Parcel influence exponent for desirability.\n        - sigma: Number of elitists for pheromone update.\n        - rho: Evaporation rate for pheromone update.\n        - width: Maximum actions to consider per expansion.\n\n        - value_function: Function to evaluate potential of partial solutions.\n        - finalize_policy: Function to complete partial solutions.\n\n        - seed: Random seed for reproducibility.\n        - time_limit: Total time limit in seconds.\n        - verbose: If True, print detailed logs.\n    \n    Returns:\n        - solution: Best solution found from the tracker (or None if failed).\n        - info: Dictionary with run statistics.\n    ';C=problem;B=swarm
	if B is _B:D=[PartialSolution(problem=C,routes=[])for A in range(n_partials)];B=PartialSolutionSwarm(solutions=D)
	E,A=_run_aco(problem=C,swarm=B,n_cutoff=n_cutoff,iterations=iterations,depth=depth,q_prob=q_prob,alpha=alpha,beta=beta,omega=omega,phi=phi,chi=chi,gamma=gamma,kappa=kappa,sigma=sigma,rho=rho,width=width,value_function=value_function,finalize_policy=finalize_policy,seed=seed,time_limit=time_limit,verbose=verbose);F=E.opt();G={_L:A[_L],_I:A[_I],_N:A[_N],_R:A[_R],_J:A[_J]};return F,G
def read_instance():
	A,B,D=map(int,sys.stdin.readline().strip().split());E=list(map(int,sys.stdin.readline().split()));F=list(map(int,sys.stdin.readline().split()));C=[[0]*(2*A+2*B+1)for C in range(2*A+2*B+1)]
	for G in range(2*A+2*B+1):H=sys.stdin.readline().strip();C[G]=list(map(int,H.split()))
	return ShareARideProblem(A,B,D,E,F,C)
def main(verbose=_A):
	F=verbose;G=read_instance();E=G.num_nodes
	if E<=100:A,B,C,D=80,16,160,8
	elif E<=250:A,B,C,D=30,10,60,6
	elif E<=500:A,B,C,D=25,5,25,5
	elif E<=1000:A,B,C,D=12,4,12,4
	else:A,B,C,D=6,3,3,3
	H,I=aco_solver(G,seed=42,verbose=F,n_partials=A,n_cutoff=B,iterations=C,width=D,time_limit=24e1);H.stdin_print(verbose=F)
if __name__=='__main__':main(verbose=_A)