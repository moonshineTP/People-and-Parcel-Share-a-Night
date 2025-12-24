_Z='average_cost'
_Y='overtime'
_X='avg_cost'
_W='done'
_V='iterations'
_U='return'
_T='N/A'
_S='elitists_count'
_R='actions_done'
_Q='dropL'
_P='pickL'
_O='serveP'
_N='iterations_completed'
_M='best_cost'
_L='actions'
_K='------------------------------'
_J='status'
_I=1.
_H='parcels'
_G='load'
_F='pos'
_E='time'
_D='ended'
_C=True
_B=None
_A=False
import sys,math,random,time
from dataclasses import dataclass
from typing import Any,Callable,Dict,List,Optional,Tuple,Union,Sequence
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
	C=verbose;A,E=0,0
	for B in seq[1:]:
		if C:print(D[A][B],end=' ')
		E+=D[A][B];A=B
	if C:print()
	return E
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
		B=verbose;print(A.problem.K)
		for(C,D)in zip(A.routes,A.route_costs):
			print(len(C));print(' '.join(map(str,C)))
			if B:print(f"// Route cost: {D}");print(_K);print()
		if B:print(f"//// Max route cost: {A.max_cost} ////")
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
			Q=E[-1];R=M>1 and E[-1]==0;S={_F:Q,_H:F,_G:H,_L:D,_D:R};L.append(S)
		return J,K,G,L,D
	def is_valid(D,verbose=_A):
		A=verbose;C=D.problem;Z,a,S=C.N,C.M,C.K
		if not len(D.routes)==len(D.states)==len(D.route_costs)==S:
			if A:print('Invalid: Mismatch in number of routes, states, or costs.')
			return _A
		if len(D.node_assignment)!=len(C.D):
			if A:print('Invalid: Mismatch in node assignment length.')
			return _A
		T=set(range(1,Z+1));U=set(range(1,a+1));O=set();P=[-1]*len(C.D);J=0;Q=0;b=0
		for B in range(S):
			G=D.routes[B];M=len(G);K=D.states[B]
			if not G or G[0]!=0:
				if A:print(f"Invalid: Route {B} does not start with depot 0.")
				return _A
			V=M>1 and G[-1]==0
			if K[_D]!=V:
				if A:print(f"Invalid: Ended state mismatch for taxi {B}.")
				return _A
			I=set();M=len(G);H=0;W=G[0];L=0
			for(N,E)in enumerate(G[1:],start=1):
				if not 0<=E<C.num_nodes:
					if A:print(f"Invalid: Node {E} out of range in route {B}.")
					return _A
				if E!=0:
					X=P[E]
					if X not in(-1,B):
						if A:print(f"Invalid: Node {E} assigned to an unintended route {X} instead of {B}.")
						return _A
					P[E]=B
				L+=C.D[W][E];W=E
				if C.is_ppick(E):
					R=C.rev_ppick(E)
					if N+1<M:
						c=G[N+1];Y=C.pserve(R)[1]
						if c!=Y:
							if A:print(f"Invalid: Passenger {R} pickup at node {E} not followed by drop at node {Y} in route {B}.")
							return _A
					J+=1;T.discard(R)
				elif C.is_pdrop(E):0
				elif C.is_lpick(E):
					F=C.rev_lpick(E)
					if F in I:
						if A:print(f"Invalid: Parcel {F} picked up multiple times in route {B}.")
						return _A
					H+=C.q[F-1]
					if H>C.Q[B]:
						if A:print(f"Invalid: Taxi {B} load {H} exceeds capacity {C.Q[B]} after picking parcel {F} in index {N}.")
						return _A
					J+=1;I.add(F);U.discard(F);O.add(F)
				elif C.is_ldrop(E):
					F=C.rev_ldrop(E)
					if F not in I:
						if A:print(f"Invalid: Parcel {F} dropped without being picked up in route {B}.")
						return _A
					H-=C.q[F-1]
					if H<0:
						if A:print(f"Invalid: Taxi {B} has negative load after dropping parcel {F}.")
						return _A
					J+=1;I.remove(F);O.discard(F)
				else:
					if N!=M-1:
						if A:print(f"Invalid: Depot node in the middle of route {B}.")
						return _A
					if H!=0 or I:
						if A:print(f"Invalid: Taxi {B} has load remaining upon returning to depot.")
						return _A
					if not V:
						if A:print(f"Invalid: Taxi {B} route does not end properly after returning to depot.")
						return _A
					J+=1
			if K[_H]!=I:
				if A:print(f"Invalid: Parcel onboard state mismatch for taxi {B}. Expected {I}, got {K[_H]}.")
				return _A
			if K[_G]!=H:
				if A:print(f"Invalid: Load state mismatch for taxi {B}. Expected {K[_G]}, got {H}.")
				return _A
			if D.route_costs[B]!=L:
				if A:print(f"Invalid: Cost state mismatch for taxi {B}. Expected {D.route_costs[B]}, got {L}.")
				return _A
			Q=max(Q,L);b+=L
		if T!=D.remaining_pass_serve:
			if A:print('Invalid: Remaining passenger serve set mismatch.')
			return _A
		if U!=D.remaining_parc_pick:
			if A:print('Invalid: Remaining parcel pick set mismatch.')
			return _A
		if O!=D.remaining_parc_drop:
			if A:print('Invalid: Remaining parcel drop set mismatch.')
			return _A
		if P!=D.node_assignment:
			if A:print('Invalid: Node assignment mismatch.')
			return _A
		if D.max_cost!=Q:
			if A:print('Invalid: Max cost mismatch.')
			return _A
		if D.num_actions!=J:
			if A:print(f"Invalid: Total actions mismatch: expected {D.num_actions}, got {J}.")
			return _A
		return _C
	def is_pending(A):return A.num_actions<A.problem.num_actions
	def is_identical(A,other):
		B=other
		if A is B:return _C
		if A.problem is not B.problem or A.num_actions!=B.num_actions:return _A
		return sorted(tuple(A)for A in A.routes)==sorted(tuple(A)for A in B.routes)
	def copy(A):return PartialSolution(problem=A.problem,routes=[A.copy()for A in A.routes]if A.routes else _B)
	def stdin_print(A,verbose=_A):
		B=verbose;print(A.problem.K)
		for(C,D)in zip(A.routes,A.route_costs):
			print(len(C));print(' '.join(map(str,C)))
			if B:print(f"// Route cost: {D}");print(_K);print()
		if B:print(f"//// Max route cost: {A.max_cost} ////")
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
		A=E.problem;G=F[_F];C=[]
		for I in E.remaining_pass_serve:J,K=A.pserve(I);D=A.D[G][J]+A.D[J][K];C.append((_O,I,D))
		for B in E.remaining_parc_pick:
			L=A.q[B-1]
			if F[_G]+L<=A.Q[H]:D=A.D[G][A.lpick(B)];C.append((_P,B,D))
		for B in F[_H]:D=A.D[G][A.ldrop(B)];C.append((_Q,B,D))
		C.sort(key=lambda x:x[2]);return C
	def check_expand(A,route_idx,kind,actid):
		E=route_idx;C=actid;B=kind;D=A.states[E];F=A.problem
		if D[_D]:return _A
		if B==_O:return C in A.remaining_pass_serve
		if B==_P:return C in A.remaining_parc_pick and D[_G]+F.q[C-1]<=F.Q[E]
		if B==_Q:return C in D[_H]
		raise ValueError(f"Unknown action kind: {B}")
	def check_return(B,route_idx):A=B.states[route_idx];return not(A[_D]or A[_H])
	def apply_extend(A,route_idx,kind,actid,inc):
		G=kind;C=actid;B=route_idx;I=A.routes[B];D=A.states[B];E=A.problem
		if D[_D]:raise ValueError(f"Cannot apply action on ended route {B}.")
		if G==_O:K,J=E.pserve(C);I.append(K);I.append(J);A.node_assignment[K]=B;A.node_assignment[J]=B;A.remaining_pass_serve.discard(C);D[_F]=J;D[_L]+=1;A.route_costs[B]+=inc;A.max_cost=max(A.max_cost,A.route_costs[B]);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions+=1;return
		elif G==_P:
			F=E.q[C-1]
			if D[_G]+F>E.Q[B]:raise ValueError(f"Taxi {B} capacity exceeded for parcel {C}.")
			H=E.lpick(C);D[_G]+=F;D[_H].add(C);A.remaining_parc_pick.discard(C);A.remaining_parc_drop.add(C)
		elif G==_Q:
			F=E.q[C-1]
			if D[_G]-F<0:raise ValueError(f"Taxi {B} load cannot be negative after dropping parcel {C}.")
			H=E.ldrop(C);D[_G]-=F;D[_H].discard(C);A.remaining_parc_drop.discard(C)
		else:raise ValueError(f"Unknown action kind: {G}")
		D[_F]=H;D[_L]+=1;I.append(H);A.node_assignment[H]=B;A.route_costs[B]+=inc;A.max_cost=max(A.max_cost,A.route_costs[B]);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions+=1
	def apply_return(A,t_idx):
		C=t_idx;D=A.routes[C];B=A.states[C]
		if B[_D]:return
		if B[_F]==0 and B[_L]>0:B[_D]=_C;return
		if B[_H]:raise ValueError(f"Taxi {C} must drop all loads before returning to depot.")
		E=A.problem.D[B[_F]][0];D.append(0);B[_F]=0;B[_L]+=1;B[_D]=_C;A.route_costs[C]+=E;A.max_cost=max(A.max_cost,A.route_costs[C]);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions+=1
	def reverse_action(A,t_idx):
		G=t_idx;F=A.routes[G];C=A.states[G]
		if len(F)<=1:raise ValueError(f"No actions to reverse for taxi {G}.")
		B=A.problem;D=F[-1]
		if B.is_pdrop(D):
			J=F.pop();I=F.pop();L=B.rev_pdrop(J)
			if B.rev_ppick(I)!=L:raise ValueError('Inconsistent route state: pdrop not preceded by corresponding ppick.')
			H=F[-1];K=B.D[H][I]+B.D[I][J];C[_F]=H;C[_L]-=1;C[_D]=_A;A.remaining_pass_serve.add(L);A.node_assignment[J]=-1;A.node_assignment[I]=-1;A.route_costs[G]-=K;A.max_cost=max(A.route_costs);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.num_actions-=1;return
		D=F.pop();H=F[-1];K=B.D[H][D];C[_F]=H;C[_L]-=1;C[_D]=_A
		if B.is_lpick(D):E=B.rev_lpick(D);C[_G]-=B.q[E-1];C[_H].discard(E);A.remaining_parc_pick.add(E);A.remaining_parc_drop.discard(E)
		elif B.is_ldrop(D):E=B.rev_ldrop(D);C[_G]+=B.q[E-1];C[_H].add(E);A.remaining_parc_pick.discard(E);A.remaining_parc_drop.add(E)
		elif D==0:0
		else:raise ValueError(f"Unexpected node type to reverse: {D}")
		A.route_costs[G]-=K;A.max_cost=max(A.route_costs);A.min_cost=min(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.node_assignment[D]=-1;A.num_actions-=1
	def is_completed(A,verbose=_A):
		B=verbose
		if A.num_actions!=A.problem.num_actions:
			if B:print(f"Not completed: current partial actions {A.num_actions} does not suffice total actions {A.problem.num_actions}.")
			return _A
		if not all(A[_D]for A in A.states):
			if B:print('Not completed: at least one route has not ended at depot.')
			return _A
		return _C
	def to_solution(A):
		if not A.is_completed(verbose=_C):print('Warning: Solution is not complete, cannot convert.');return
		if not A.is_valid(verbose=_C):print('Warning: Solution is not valid, cannot convert.');return
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
	def stats(A):A.update();return{'num_partials':A.num_partials,'min_cost':A.min_cost,'max_cost':A.max_cost,_X:A.avg_cost}
	def copy(A):B=[A.copy()for A in A.partial_lists];return PartialSolutionSwarm(solutions=B)
def weighted(kind,inc,pweight=.7):
	if kind==_O:return pweight*inc
	return inc+1
Action=Tuple[int,str,int,int]
def action_weight(action,pweight=.7):A=action;return weighted(A[1],A[3],pweight)
def softmax_weighter(incs,t):
	A=incs;B,E=min(A),max(A);C=E-B
	if C<1e-06:return[_I]*len(A)
	D=[]
	for F in A:G=(F-B)/C;D.append((_I-G+.1)**(_I/t))
	return D
Action=Tuple[int,str,int,int]
def balanced_scorer(partial,sample_size=8,w_std=.15,seed=_B):
	B=sample_size;A=partial;E=random.Random(seed);C=sorted(A.route_costs)
	if len(C)==1:return A.max_cost
	D=E.choices(C,k=B);F=sum(D)/B;G=sum((A-F)**2 for A in D)/B;H=G**.5;return A.max_cost+w_std*H
def check_general_action(partial,action):
	A=partial;B,C,D,E=action
	if C==_U:return A.check_return(B)
	return A.check_expand(B,C,D)
def apply_general_action(partial,action):
	A=partial;B,C,D,E=action
	if C==_U:A.apply_return(B)
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
		if A.num_actions<B.num_expansions:print('[Warning] No feasible actions found before closing depth');raise RuntimeError('Premature routes not covering all nodes.')
		S=[]
		for C in E:
			Y=A.states[C]
			if A.check_return(C):Z=B.D[Y[_F]][0];S.append((C,_U,0,Z))
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
	E=verbose;B=partial;A=route_idx;G=random.Random(seed);C=0
	for N in range(steps):
		H=B.states[A]
		if H[_D]:break
		D=B.possible_expand(A)
		if not D:
			if E:print(f"[Repair] Route {A} has no feasible actions, return to depot.")
			B.apply_return(A);C+=1;break
		I=[weighted(A,B)for(A,C,B)in D];J=softmax_weighter(I,T);F=sample_from_weight(G,J);K,L,M=D[F];B.apply_extend(A,K,L,M);C+=1
		if E:print(f"[Repair] Route {A} select action {D[F]}")
	if E:print(f"[Repair] Route {A} finished building, added {C} actions.")
	return B,C
def repair_operator(partial,repair_proba=_B,steps=_B,T=_I,seed=_B,verbose=_A):
	D=verbose;C=repair_proba;B=steps;A=partial;H=random.Random(seed)
	if C is _B:C=_I
	if B is _B:B=10**9
	K=list(range(A.problem.K));E=A.problem.K;L=round(C*E+.5);I=min(E,max(1,L));M=H.sample(K,I);F=0;J=[_A]*E
	for G in M:
		A,N=repair_one_route(partial=A,route_idx=G,steps=B,T=T,seed=H.randint(0,1000000),verbose=D);F+=N;J[G]=_C
		if D:print(f"[Repair]: Repairing route {G} with up to {B} steps.")
	if D:print();print('[Repair] Operator completed.');print(f"Total routes repaired: {I};");print(f"Total actions added: {F}.");print(_K);print()
	A.stdin_print();return A,J,F
from typing import List,Tuple,Optional,Union
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
	if verbose:print(f"[Destroy] Route {route_idx}: removed {D} actions.")
	return C,D
def destroy_operator(sol,destroy_proba,destroy_steps,seed=_B,t=_I,verbose=_A):
	I=verbose;A=sol;N=random.Random(seed);B=A.problem.K;D=[A[:]for A in A.routes];O=A.route_costs;P=round(destroy_proba*B+.5);Q=min(B,max(1,P));R=softmax_weighter(O,t=t);E=[];F=list(range(B));J=R[:]
	for U in range(Q):
		if not F:break
		G=sample_from_weight(N,J);E.append(F[G]);F.pop(G);J.pop(G)
	K=[_A]*B;H=0
	for C in E:
		L=D[C]
		if len(L)<=2:continue
		S,M=destroy_one_route(A.problem,L,C,steps=destroy_steps,verbose=I)
		if M>0:D[C]=S;K[C]=_C;H+=M
	if I:print();print('[Destroy] Operation complete.');print(f"[Destroy] Destroyed {len(E)} routes, removed {H} nodes total.");print(_K);print()
	T=PartialSolution(problem=A.problem,routes=D);return T,K,H
def greedy_solver(problem,partial=_B,verbose=_A):
	B=verbose;A=partial;E=time.time()
	if A is _B:A=PartialSolution(problem=problem)
	C=0;F=A.num_actions
	while A.is_pending():
		C+=1;G=enumerate_actions_greedily(A,1)
		if not G:
			if B:print('[Greedy] [Error] The partial has no feasible actions available.')
			return _B,{_V:C,_E:time.time()-E,_R:A.num_actions-F,_J:'error'}
		H=G[0];apply_general_action(A,H)
		if B:J,K,L,M=H;print(f"[Greedy] [Depth {A.num_actions}] Taxi {J} extended route with action {K} on passenger/parcel {L}")
	D=A.to_solution();I={_V:C,_E:time.time()-E,_R:A.num_actions-F,_J:_W}
	if B:print();print('[Greedy] Completed.');print(f"[Greedy] Solution max cost: {D.max_cost if D else _T}");print(f"[Greedy] Time taken: {I[_E]:.4f} seconds");print(_K);print()
	return D,I
def iterative_greedy_solver(problem,partial=_B,iterations=10000,destroy_proba=.53,destroy_steps=13,destroy_t=1.3,rebuild_proba=.29,rebuild_steps=3,rebuild_t=1.2,time_limit=3e1,seed=_B,verbose=_A):
	I=seed;H=partial;G=problem;D=verbose;J=time.time();L=J+time_limit;W=random.Random(I)
	if H is _B:H=PartialSolution(problem=G,routes=[])
	A,X=greedy_solver(G,partial=H,verbose=D)
	if not A:return _B,{_E:time.time()-J,_J:'error'}
	C=A.max_cost
	if D:print(f"[Iterative Greedy] [Iter 0] initial best cost: {C}")
	E=X[_R];M=C;N=0;O=0;P=0;Q=_W;F=0
	for R in range(1,iterations+1):
		if L and time.time()>=L:Q=_Y;break
		S=_B if I is _B else 2*I+98*R;Y,Z,T=destroy_operator(A,destroy_proba,destroy_steps,seed=S,t=destroy_t);O+=T;E+=T;K=Y
		for(a,b)in enumerate(Z):
			if not b:continue
			if W.random()>rebuild_proba:continue
			K,U=repair_one_route(K,route_idx=a,steps=rebuild_steps,T=rebuild_t,seed=S);P+=U;E+=U
		B,c=greedy_solver(G,partial=K,verbose=_A);M+=B.max_cost if B else 0;E+=c[_R];F+=1 if B else 0
		if B and B.max_cost<C:
			A=B;C=B.max_cost;N+=1
			if D:print(f"[Iterative Greedy] [Iter {R}] improved best to {C}")
	V=time.time()-J;d={_V:F,_R:E,'improvements':N,'actions_destroyed':O,'actions_rebuilt':P,_Z:M/(F+1),_E:V,_J:Q}
	if D:print();print(f"[Iterative Greedy] Finished after {F} iterations.");print(f"[Iterative Greedy] Best solution max cost: {A.max_cost if A else _T}.");print(f"[Iterative Greedy] Time taken: {V:.4f} seconds.");print(_K);print()
	return A,d
import heapq
from typing import List,Optional,Tuple,Dict
from typing import Callable,Union,Sequence,List,Tuple
import bisect
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
Request=Tuple[int,int,str]
ActionNode=Tuple[int,int]
ValueFunction=Callable
FinalizePolicy=Callable
def _default_value_function(partial,perturbed_samples=8,seed=_B):A=partial;C,B=C,B=iterative_greedy_solver(A.problem,A,iterations=perturbed_samples,time_limit=.1,seed=seed);return B[_Z]
def _default_finalize_policy(partial,iterations=1000,seed=_B):
	A=partial;B,C=iterative_greedy_solver(A.problem,A,iterations=iterations,time_limit=3.,seed=seed,verbose=_A)
	if not B:return
	return B
@dataclass
class SolutionTracker:
	best_solution:Optional[Solution]=_B;best_cost:int=10**18;worst_cost:int=-1;total_cost:int=0;count:int=0
	def update(B,source):
		A=source
		if isinstance(A,Solution):B._update_from_solution(A)
		elif isinstance(A,PartialSolutionSwarm):B._update_from_swarm(A)
		elif isinstance(A,list):B._update_from_list(A)
	def _update_from_solution(A,solution):
		C=solution;B=C.max_cost;A.count+=1;A.total_cost+=B;A.worst_cost=max(A.worst_cost,B)
		if B<A.best_cost:A.best_cost=B;A.best_solution=C
	def _update_from_swarm(C,swarm):
		for A in swarm.partial_lists:
			if not A.is_completed():continue
			B=A.to_solution()
			if not B:continue
			C._update_from_solution(B)
	def _update_from_list(B,solutions):
		for A in solutions:
			if A is _B:continue
			B._update_from_solution(A)
	def stats(A):B=A.total_cost/A.count if A.count>0 else .0;return{_M:A.best_cost,'worst_cost':A.worst_cost,_X:B,'count':A.count}
	def opt(A):return A.best_solution
class PheromoneMatrix:
	def __init__(A,problem,sigma,rho,init_cost):A.size=problem.num_nodes;A.sigma=sigma;A.rho=rho;A.tau_0=_I/(rho*init_cost);A.tau_max=2*A.tau_0;A.tau_min=A.tau_0/1e1;A.tau=[[A.tau_0 for B in range(A.size)]for B in range(A.size)]
	def get(A,prev,curr):return A.tau[prev[1]][curr[0]]
	def update(A,swarm,opt):
		D=opt
		def F(partial):
			A=partial;B=[]
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
		D,B,E,G=action;F=A.problem.Q[D];H=partial.states[D];C=H[_G]
		if B==_P:C+=A.problem.q[E-1]
		if B==_Q:C-=A.problem.q[E-1]
		I=(1+weighted(B,G))**A.chi;J=A.saving_matrix[prev[1]][curr[0]];K=2-int(B=='pickP');L=(1+A.gamma*(F-C)/F)*A.kappa;return J/I*K*L
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
			M=G[_F];N=K.nearest_actions[M]
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
			F,B,C,G=action;D=E.partial.problem
			if B==_O:return D.pserve(C)
			elif B==_P:A=D.lpick(C);return A,A
			elif B==_Q:A=D.ldrop(C);return A,A
			else:return 0,0
		def _collect_actions(F):
			B=F.partial;C=F.width
			if B.num_actions>=B.problem.num_expansions:A=enumerate_actions_greedily(B,C);D=action_weight(A[0]);return[(D/action_weight(A),A)for A in A]
			A=F.cache.query(B,C)
			if len(A)<C:A=enumerate_actions_greedily(B,C)[:C]
			G=[];H=_A;I=.0;J=A[0];D=action_weight(J);G.append((D,J))
			for(L,E)in zip(A[:-1],A[1:]):
				M=L[3];N=E[3]
				if not H and N>M:H=_C;O=action_weight(E);I=O+_I
				K=action_weight(E)
				if H:K+=I
				G.append((K,E))
			P=[(D/A,B)for(A,B)in G];return P
		def _compute_log_proba(A,tau,eta,fit,action):B=action;G=B[0];H=A.partial.states[G];I=H[_F];E=10**18,I;F=A._get_action_node(B);C=tau.get(E,F);D=eta.get(E,F,A.partial,B);C=max(C,1e-300);D=max(D,1e-300);J=+A.alpha*math.log(C)+A.beta*math.log(D)+A.omega*math.log(fit);return J
		def sample_action(A,tau,eta,rng):
			B=A._collect_actions()
			if not B:return
			C=[]
			for(G,H)in B:I=A._compute_log_proba(tau,eta,G,H);C.append(I)
			J=max(C);F=[math.exp(A-J)for A in C];K=sum(F);D=[A/K for A in F];E:0
			if rng.random()<A.q_prob:E=D.index(max(D))
			else:E=sample_from_weight(rng,D)
			return B[E][1]
	def __init__(A,partial,cache,tau,eta,alpha,beta,omega,q_prob,width,rng):G=width;F=q_prob;E=omega;D=alpha;C=cache;B=partial;A.problem=B.problem;A.partial=B;A.cache=C;A.tau=tau;A.eta=eta;A.alpha=D;A.beta=beta;A.omega=E;A.q_prob=F;A.width=G;A.rng=rng;A.sampler=Ant.ProbaExpandSampler(partial=A.partial,cache=C,alpha=D,beta=beta,omega=E,q_prob=F,width=G)
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
		G=[A.max_actions//5,A.max_actions//2,A.max_actions*9//10]
		for B in range(A.depth):
			if A.tle():
				if A.verbose:print('[ACO] Time limit reached, skipping this iteration.')
				return A.swarm
			if A.verbose:
				if B in G or B%100==0:C=[A.max_cost for A in A.swarm.partial_lists];D=[A.num_actions for A in A.swarm.partial_lists];print(f"[ACO] [Iteration {B}] Partial cost range: {min(C):.3f} - {max(C):.3f}, Depth range: {min(D)} - {max(D)}, Time_elapsed={time.time()-A.start_time:.2f}s.")
			if not A.expand():
				if A.verbose:print('[ACO] All ants have completed their solutions.')
				break
			A.update()
		if A.verbose:H=sum(1 for A in A.swarm.partial_lists if A.is_completed());E=A.swarm.opt();F=A.lfunc.opt();print(f"[ACO] Finished all depth.\nComplete solutions found: {H}/{A.num_ants}.\nRun best cost: {E.max_cost if E else _T}, Opt cost: {F.max_cost if F else _T}.")
		return A.swarm
class SwarmTracker:
	def __init__(A,swarm,value_function,finalize_policy,seed=_B):C=value_function;B=swarm;A.seed=seed;A.frontier_swarm=[A.copy()for A in B.partial_lists];A.num_partials=B.num_partials;A.frontier_fitness=[C(A)for A in A.frontier_swarm];(A.finals):0;A.is_finalized=_A;A.value_function=C;A.finalize_policy=finalize_policy
	def update(A,source):
		for(B,C)in enumerate(source.partial_lists):
			D=A.value_function(C,seed=A.seed+10*B if A.seed else _B)
			if D<A.frontier_fitness[B]:A.frontier_swarm[B]=C.copy();A.frontier_fitness[B]=D
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
	K=time_limit;J=swarm;H=depth;G=n_cutoff;D=problem;B=iterations;A=verbose;L=time.time()
	if H is _B:H=D.num_actions
	if A:print('[ACO] [Init] Estimating costs from initial greedy solver...')
	M,Y=greedy_solver(D,_B,_A);N=M.max_cost
	if A:print(f"[ACO] [Init] Greedy solution cost: {N:.3f}")
	if A:print('[ACO] [Init] Initializing nearest expansion cache...')
	S=NearestExpansionCache(D,n_nearest=5)
	if A:print('[ACO] [Init] Initializing matrices...')
	T=PheromoneMatrix(D,sigma=sigma,rho=rho,init_cost=N);U=DesirabilityMatrix(D,phi,chi,gamma,kappa)
	if A:print('[ACO] [Init] Initializing trackers...')
	I=SolutionTracker();I.update(M);E=SwarmTracker(swarm=J,value_function=value_function,finalize_policy=finalize_policy);O=B;P=_W
	for C in range(B):
		if time.time()-L>=.75*K:
			O=C;P=_Y
			if A:print(f"[ACO] Time limit approaching, stopping at run {C+1}/{B}.")
			break
		if A:print(f"[ACO] [Run {C+1}/{B}] Starting the population run...")
		V=AntPopulation(swarm=J,cache=S,tau=T,eta=U,lfunc=I,alpha=alpha,beta=beta,omega=omega,q_prob=q_prob,width=width,depth=H,time_limit=K,seed=hash(seed+10*C)if seed else _B,verbose=A)
		if A:print(f"[ACO] [Run {C+1}/{B}] Running the ant population")
		Q=V.run()
		if A:print(f"[ACO] [Run {C+1}/{B}] Updating swarm tracker")
		E.update(Q);I.update(Q)
	if A:print(f"[ACO] Finalizing top {G} partial into solutions...")
	E.finalize(G);W=time.time()-L;R=E.opt(cutoff=G);X=R.max_cost if R else float('inf');F={_N:O,_M:X,_S:E.num_partials,_E:W,_J:P}
	if A:print(f"[ACO] The run finished. iterations_completed={F[_N]}, Best_cost={F[_M]:.3f}, Time={F[_E]:.3f}s.")
	return E,F
def aco_enumerator(problem,swarm=_B,n_partials=50,n_cutoff=20,n_return=5,iterations=10,depth=_B,q_prob=.75,alpha=1.2,beta=1.4,omega=4,phi=.5,chi=1.5,gamma=.4,kappa=2.,sigma=10,rho=.55,width=8,value_function=_default_value_function,finalize_policy=_default_finalize_policy,seed=_B,time_limit=3e1,verbose=_A):
	H='solutions_found';G=verbose;F=n_cutoff;E=problem;C=swarm
	if C is _B:I=[PartialSolution(problem=E,routes=[])for A in range(n_partials)];C=PartialSolutionSwarm(solutions=I)
	J,A=_run_aco(problem=E,swarm=C,n_cutoff=F,iterations=iterations,depth=depth,q_prob=q_prob,alpha=alpha,beta=beta,omega=omega,phi=phi,chi=chi,gamma=gamma,kappa=kappa,sigma=sigma,rho=rho,width=width,value_function=value_function,finalize_policy=finalize_policy,seed=seed,time_limit=time_limit,verbose=G);B=J.top(k=n_return,cutoff=F);D={_N:A[_N],_E:A[_E],_M:A[_M],H:len(B),_S:A[_S],_J:A[_J]}
	if G:print();print('[ACO] Enumeration complete.');print(f"[ACO] Total solutions found: {D[H]}.");print(f"[ACO] Solution costs range: {B[0].max_cost:.3f} - {B[-1].max_cost:.3f}.");print(f"[ACO] Total time: {D[_E]:.3f}s");print(_K);print()
	return B,D
def aco_solver(problem,swarm=_B,n_partials=20,n_cutoff=10,iterations=10,depth=_B,q_prob=.75,alpha=1.2,beta=1.4,omega=4,phi=.5,chi=1.5,gamma=.4,kappa=2.,sigma=10,rho=.55,width=5,value_function=_default_value_function,finalize_policy=_default_finalize_policy,seed=_B,time_limit=3e1,verbose=_A):
	E=verbose;D=problem;B=swarm
	if B is _B:G=[PartialSolution(problem=D,routes=[])for A in range(n_partials)];B=PartialSolutionSwarm(solutions=G)
	H,A=_run_aco(problem=D,swarm=B,n_cutoff=n_cutoff,iterations=iterations,depth=depth,q_prob=q_prob,alpha=alpha,beta=beta,omega=omega,phi=phi,chi=chi,gamma=gamma,kappa=kappa,sigma=sigma,rho=rho,width=width,value_function=value_function,finalize_policy=finalize_policy,seed=seed,time_limit=time_limit,verbose=E);C=H.opt();F={_N:A[_N],_E:A[_E],_M:A[_M],_S:A[_S],_J:A[_J]}
	if E:
		print();print('[ACO] Solver complete.')
		if C is not _B:print(f"[ACO] Best solution cost: {C.max_cost}")
		else:print('[ACO] No valid solution found.')
		print(f"[ACO] Total time: {F[_E]:.3f}s");print(_K);print()
	return C,F
def read_instance():
	A,B,D=map(int,sys.stdin.readline().strip().split());E=list(map(int,sys.stdin.readline().split()));F=list(map(int,sys.stdin.readline().split()));C=[[0]*(2*A+2*B+1)for C in range(2*A+2*B+1)]
	for G in range(2*A+2*B+1):H=sys.stdin.readline().strip();C[G]=list(map(int,H.split()))
	return ShareARideProblem(A,B,D,E,F,C)
def main(verbose=_A):
	F=verbose;G=read_instance();E=G.num_nodes
	if E<=100:A,B,C,D=150,25,150,6
	elif E<=250:A,B,C,D=60,15,60,5
	elif E<=500:A,B,C,D=25,10,25,4
	elif E<=1000:A,B,C,D=12,4,8,3
	else:A,B,C,D=6,2,3,2
	H,I=aco_solver(G,seed=42,verbose=F,n_partials=A,n_cutoff=B,iterations=C,width=D,time_limit=24e1);H.stdin_print(verbose=F)
if __name__=='__main__':main(verbose=_A)