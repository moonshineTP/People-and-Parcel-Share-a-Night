_c='----------------'
_b='elitists_count'
_a='first'
_Z='-inf'
_Y='return'
_X='runs_completed'
_W='N/A'
_V='------------------------------'
_U='dropP'
_T='best_cost'
_S='inf'
_R='iterations'
_Q='dropL'
_P='pickL'
_O='pickP'
_N='actions'
_M='actions_evaluated'
_L='time'
_K=1.
_J='route'
_I='load'
_H='parcels'
_G='cost'
_F='pos'
_E='passenger'
_D='ended'
_C=True
_B=None
_A=False
import sys,random,time,math
from typing import Optional,List,Tuple,Dict,Any,Union,Callable
from dataclasses import dataclass
Action=Tuple[int,str,int,int]
class ShareARideProblem:
	def __init__(A,N,M,K,parcel_qty,vehicle_caps,dist,coords=_B):A.N=N;A.M=M;A.K=K;A.q=list(parcel_qty);A.Q=list(vehicle_caps);A.D=[A[:]for A in dist];A.num_nodes=2*N+2*M+1;A.num_requests=N+M;A.num_actions=2*N+2*M+K;A.ppick=lambda i:i;A.pdrop=lambda i:N+M+i;A.parc_pick=lambda j:N+j;A.parc_drop=lambda j:2*N+M+j;A.rev_ppick=lambda i:i;A.rev_pdrop=lambda n:n-(N+M);A.rev_parc_pick=lambda n:n-N;A.rev_parc_drop=lambda n:n-(2*N+M);A.is_ppick=lambda x:1<=x<=N;A.is_pdrop=lambda x:N+M+1<=x<=2*N+M;A.is_parc_pick=lambda x:N+1<=x<=N+M;A.is_parc_drop=lambda x:2*N+M+1<=x<=2*(N+M);A.coords=coords
	def is_valid(A):
		if len(A.q)!=A.M:return _A
		if len(A.Q)!=A.K:return _A
		if len(A.D)!=A.num_nodes:return _A
		if not all(len(B)==A.num_nodes for B in A.D):return _A
		if A.coords is not _B and len(A.coords)!=A.num_nodes:return _A
		return _C
	def copy(A):return ShareARideProblem(A.N,A.M,A.K,list(A.q),list(A.Q),[A[:]for A in A.D])
	def stdin_print(A):
		print(A.N,A.M,A.K);print(*A.q);print(*A.Q)
		for B in A.D:print(*B)
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
	def is_valid(H):
		A=H.problem;L=A.K
		if len(H.routes)!=L:return _A
		for(M,I)in enumerate(H.routes):
			if not(I[0]==0 and I[-1]==0):return _A
			E=set();F=set();G=0;J=set();K=set()
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
					if B in K or B in F:return _A
					G+=A.q[B-1]
					if G>A.Q[M]:return _A
					K.add(B);F.add(B)
				elif A.is_parc_drop(C):
					B=A.rev_parc_drop(C)
					if B not in F:return _A
					G-=A.q[B-1];F.remove(B)
			if E:return _A
			if G!=0:return _A
		return _C
	def stdin_print(A,verbose=_A):
		B=verbose;print(A.problem.K)
		for(C,D)in zip(A.routes,A.route_costs):
			print(len(C));print(' '.join(map(str,C)))
			if B:print(f"// Route cost: {D}");print(_c)
		if B:print(f"//// Max route cost: {A.max_cost} ////")
class PartialSolution:
	def __init__(A,problem,routes):C=routes;B=problem;A.problem=B;A.routes=A._init_routes(C);A.route_costs=A._init_costs(C);A.max_cost=max(A.route_costs);A.avg_cost=sum(A.route_costs)/B.K;A.node_assignment=A._init_node_assignment();A.remaining_pass_pick,A.remaining_pass_drop,A.remaining_parc_pick,A.remaining_parc_drop,A.route_states=A._init_states();A.num_actions=sum(len(A)-1 for A in A.routes)
	def _init_routes(D,routes):
		A=routes;B=D.problem.K
		if not A:return[[0]for A in range(B)]
		if len(A)!=B:raise ValueError(f"Expected {B} routes, got {len(A)}.")
		for C in A:
			if C is _B:raise ValueError('One route cannot be null')
			elif not C or C[0]!=0:raise ValueError('Each route must start at depot 0.')
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
			P=D[-1];Q=len(D)>1 and D[-1]==0;R={_J:D,_F:P,_G:H.route_costs[O],_I:K,_E:F,_H:G,_N:len(D)-1,_D:Q};N.append(R)
		return L,I,M,J,N
	def is_valid(B):
		A=B.problem;W,X,Q=A.N,A.M,A.K
		if not len(B.routes)==len(B.route_states)==len(B.route_costs)==Q:return _A
		if len(B.node_assignment)!=len(A.D):return _A
		R=set(range(1,W+1));M=set();S=set(range(1,X+1));N=set();O=[-1]*len(A.D);T=0;P=0;Y=0
		for I in range(Q):
			D=B.routes[I];F=B.route_states[I]
			if not D or D[0]!=0:return _A
			if F[_J]!=D:return _A
			if F[_F]!=D[-1]:return _A
			if F[_N]!=len(D)-1:return _A
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
			if F[_H]!=J:return _A
			if F[_I]!=L:return _A
			if F[_G]!=K or B.route_costs[I]!=K:return _A
			T+=len(D)-1;P=max(P,K);Y+=K
		if R!=B.remaining_pass_pick:return _A
		if M!=B.remaining_pass_drop:return _A
		if S!=B.remaining_parc_pick:return _A
		if N!=B.remaining_parc_drop:return _A
		if O!=B.node_assignment:return _A
		if B.max_cost!=P:return _A
		if B.num_actions!=T:return _A
		return _C
	def is_identical(A,other):
		B=other
		if A is B:return _C
		if A.problem is not B.problem:return _A
		if A.num_actions!=B.num_actions:return _A
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
		B=verbose;print(A.problem.K)
		for(C,D)in zip(A.routes,A.route_costs):
			print(len(C));print(' '.join(map(str,C)))
			if B:print(f"// Route cost: {D}");print(_c)
		if B:print(f"//// Max route cost: {A.max_cost} ////")
	def possible_actions(G,t_idx):
		I=t_idx;C=G.route_states[I]
		if C[_D]:return[]
		A=G.problem;H=C[_F];D=[]
		if C[_E]==0:
			for F in list(G.remaining_pass_pick):B=A.D[H][A.ppick(F)];D.append((_O,F,B))
		else:F=C[_E];B=A.D[H][A.pdrop(F)];D.append((_U,F,B))
		for E in list(G.remaining_parc_pick):
			J=A.q[E-1]
			if C[_I]+J<=A.Q[I]:B=A.D[H][A.parc_pick(E)];D.append((_P,E,B))
		for E in list(C[_H]):B=A.D[H][A.parc_drop(E)];D.append((_Q,E,B))
		D.sort(key=lambda x:x[2]);return D
	def check_action(D,t_idx,kind,node_idx):
		E=t_idx;B=node_idx;A=kind;C=D.route_states[E];F=D.problem
		if C[_D]:return _A
		if A==_O:return C[_E]==0 and B in D.remaining_pass_pick
		if A==_U:return C[_E]==B
		if A==_P:return B in D.remaining_parc_pick and C[_I]+F.q[B-1]<=F.Q[E]
		if A==_Q:return B in C[_H]
		raise ValueError(f"Unknown action kind: {A}")
	def check_return(B,t_idx):A=B.route_states[t_idx];return not(A[_D]or A[_E]!=0 or A[_H])
	def apply_action(C,t_idx,kind,node_idx,inc):
		G=kind;D=t_idx;A=node_idx;B=C.route_states[D]
		if B[_D]:raise ValueError(f"Cannot apply action on ended route {D}.")
		E=C.problem
		if G==_O:
			if B[_E]!=0:raise ValueError(f"Taxi {D} already has passenger {B[_E]}.")
			F=E.ppick(A);B[_E]=A;C.remaining_pass_pick.discard(A);C.remaining_pass_drop.add(A)
		elif G==_U:
			if B[_E]!=A:raise ValueError(f"Taxi {D} is not carrying passenger {A}.")
			F=E.pdrop(A);B[_E]=0;C.remaining_pass_drop.discard(A)
		elif G==_P:
			H=E.q[A-1]
			if B[_I]+H>E.Q[D]:raise ValueError(f"Taxi {D} capacity exceeded for parcel {A}.")
			F=E.parc_pick(A);B[_I]+=H;B[_H].add(A);C.remaining_parc_pick.discard(A);C.remaining_parc_drop.add(A)
		elif G==_Q:
			if A not in B[_H]:raise ValueError(f"Taxi {D} does not carry parcel {A}.")
			F=E.parc_drop(A);B[_I]-=E.q[A-1];B[_H].discard(A);C.remaining_parc_drop.discard(A)
		else:raise ValueError(f"Unknown action kind: {G}")
		B[_J].append(F);B[_G]+=inc;B[_F]=F;B[_N]+=1;C.node_assignment[F]=D;C.route_costs[D]=B[_G];C.max_cost=max(C.max_cost,B[_G]);C.avg_cost=sum(C.route_costs)/C.problem.K;C.num_actions+=1
	def apply_return(B,t_idx):
		C=t_idx;A=B.route_states[C]
		if A[_D]:return
		if A[_F]==0 and len(A[_J])>1:A[_D]=_C;return
		if A[_E]!=0 or A[_H]:raise ValueError(f"Taxi {C} must drop all loads before returning to depot.")
		A[_G]+=B.problem.D[A[_F]][0];A[_J].append(0);A[_F]=0;A[_N]+=1;A[_D]=_C;B.route_costs[C]=A[_G];B.max_cost=max(B.max_cost,A[_G]);B.avg_cost=sum(B.route_costs)/B.problem.K;B.num_actions+=1
	def reverse_action(A,t_idx):
		G=t_idx;B=A.route_states[G]
		if len(B[_J])<=1:raise ValueError(f"No actions to reverse for taxi {G}.")
		C=B[_J].pop();H=B[_J][-1];I=A.problem.D[H][C];B[_G]-=I;B[_F]=H;B[_N]-=1;B[_D]=_A;D=A.problem
		if D.is_ppick(C):F=D.rev_ppick(C);B[_E]=0;A.remaining_pass_pick.add(F);A.remaining_pass_drop.discard(F)
		elif D.is_pdrop(C):F=D.rev_pdrop(C);B[_E]=F;A.remaining_pass_pick.discard(F);A.remaining_pass_drop.add(F)
		elif D.is_parc_pick(C):E=D.rev_parc_pick(C);B[_I]-=D.q[E-1];B[_H].discard(E);A.remaining_parc_pick.add(E);A.remaining_parc_drop.discard(E)
		elif D.is_parc_drop(C):E=D.rev_parc_drop(C);B[_I]+=D.q[E-1];B[_H].add(E);A.remaining_parc_pick.discard(E);A.remaining_parc_drop.add(E)
		else:B[_D]=_A
		A.route_costs[G]=B[_G];A.max_cost=max(A.route_costs);A.avg_cost=sum(A.route_costs)/A.problem.K;A.node_assignment[C]=-1;A.num_actions-=1
	def is_complete(A):return A.num_actions==A.problem.num_actions and all(A[_D]for A in A.route_states)
	def to_solution(A):
		if not A.is_complete():print('Cannot convert to Solution: not all routes have ended at depot.');return
		B=Solution(problem=A.problem,routes=A.routes,route_costs=A.route_costs)
		if B is _B or not B.is_valid():print('Warning: Converted solution is not valid.')
		return B
	@staticmethod
	def from_solution(sol):A=[A.copy()for A in sol.routes];return PartialSolution(problem=sol.problem,routes=A)
class PartialSolutionSwarm:
	def __init__(A,solutions):
		B=solutions
		if not B:raise ValueError('Solutions list cannot be empty.')
		A.problem=B[0].problem;A.num_partials=len(B);A.partial_lists=B;A.partial_num_actions=[A.num_actions for A in B];A.costs=[A.max_cost for A in B];A.min_cost=min(A.costs);A.max_cost=max(A.costs);A.avg_cost=sum(A.max_cost for A in B)/len(B);A.best_partial=min(B,key=lambda s:s.max_cost)
	def apply_action_one(A,sol_idx,t_idx,kind,node_idx,inc):
		C=sol_idx;B=A.partial_lists[C];B.apply_action(t_idx,kind,node_idx,inc);A.partial_num_actions[C]=B.num_actions;A.costs[C]=B.max_cost;A.min_cost=min(A.costs);A.max_cost=max(A.costs);A.avg_cost=sum(A.costs)/len(A.costs)
		if B.max_cost==A.min_cost:A.best_partial=B
	def apply_return_to_depot_one(A,sol_idx,t_idx):
		C=sol_idx;B=A.partial_lists[C];B.apply_return(t_idx);A.partial_num_actions[C]=B.num_actions;A.costs[C]=B.max_cost;A.min_cost=min(A.costs);A.max_cost=max(A.costs);A.avg_cost=sum(A.costs)/len(A.costs)
		if B.max_cost==A.min_cost:A.best_partial=B
	def copy(A):B=[A.copy()for A in A.partial_lists];return PartialSolutionSwarm(solutions=B)
	def opt(E):
		B=10**18;C=_B
		for D in E.partial_lists:
			if D.is_complete():
				A=D.to_solution()
				if A and A.max_cost<B:B=A.max_cost;C=A
		return C
from typing import Sequence
def sample_from_weight(rng,weights):
	A=weights;C=sum(A)
	if C<1e-10:B=rng.randrange(len(A))
	else:
		E=rng.random()*C;D=.0;B=0
		for(F,G)in enumerate(A):
			D+=G
			if E<=D:B=F;break
	return B
def softmax_weighter(incs,t):
	A=incs;B,E=min(A),max(A);C=E-B
	if C<1e-06:return[_K]*len(A)
	D=[]
	for F in A:G=(F-B)/C;D.append((_K-G+.1)**(_K/t))
	return D
def repair_one_route(partial,route_idx,steps,T=_K,seed=42,verbose=_A):
	D=verbose;B=partial;A=route_idx;G=random.Random(seed);E=0
	for O in range(steps):
		H=B.route_states[A]
		if H[_D]:break
		C=B.possible_actions(A)
		if D:print(f"[build] route {A} available actions: {C}")
		if not C:
			if D:print(f"[build] route {A} has no feasible actions, ending.")
			B.apply_return(A);E+=1;break
		I=[A[2]for A in C];J=softmax_weighter(I,T);F=sample_from_weight(G,J);K,L,M=C[F]
		if D:print(f"[build] route {A} selected action: {C[F]}")
		B.apply_action(A,K,L,M);E+=1
	if D:print(f"[build] route {A} finished building, added {E} nodes.")
	N=[B==A for B in range(B.problem.K)];return B,N,E
def repair_operator(partial,repair_proba,steps,T=_K,seed=42,verbose=_A):
	G=steps;B=verbose;A=partial;H=random.Random(seed);J=[A for A in range(A.problem.K)];C=A.problem.K;K=round(repair_proba*C+.5);I=min(C,max(1,K));L=H.sample(J,I);D=0;E=[_A]*C
	for F in L:
		A,E,M=repair_one_route(partial=A,route_idx=F,steps=G,T=T,seed=H.randint(0,1000000),verbose=B);D+=M;E[F]=_C
		if B:print(f"[Repair]: Repairing route {F} with up to {G} steps.")
	if B:print();print('[Repair] Operator completed.');print(f"Total routes repaired: {I};");print(f"Total nodes added: {D}.");print(_V);print()
	return A,E,D
from typing import List,Tuple,Optional
def destroy_one_route(route,route_idx,steps=10,verbose=_A):
	D=route;A=D[:-1];B=min(steps,max(0,len(A)-1))
	if B<=0:return D[:]
	E=len(A)-B;C=A[:E]
	if not C:C=[0]
	if verbose:print(f"[Destroy] Route {route_idx}: removed last {B} nodes.")
	return C
def destroy_operator(sol,destroy_proba,destroy_steps,seed=_B,t=_K,verbose=_A):
	J=verbose;B=sol;N=random.Random(seed);A=[A[:]for A in B.routes];O=B.route_costs;E=[_A]*len(A);C=0
	if not A:return PartialSolution(problem=B.problem,routes=A),E,C
	P=round(destroy_proba*len(A)+.5);Q=min(B.problem.K,max(1,P));R=softmax_weighter(O,t=t);F=[];G=list(range(B.problem.K));K=R[:]
	for T in range(Q):
		if not G:break
		H=sample_from_weight(N,K);F.append(G[H]);G.pop(H);K.pop(H)
	for D in F:
		I=A[D]
		if len(I)<=2:continue
		L=destroy_one_route(I,D,steps=destroy_steps,verbose=J);M=max(0,len(I)-len(L))
		if M>0:A[D]=L;E[D]=_C;C+=M
	S=PartialSolution(problem=B.problem,routes=A)
	if J:print();print('[Destroy] Operation complete.');print(f"[Destroy] Destroyed {len(F)} routes, removed {C} nodes total.");print(_V);print()
	return S,E,C
def greedy_balanced_solver(problem,partial=_B,verbose=_A):
	G=verbose;A=partial
	if A is _B:A=PartialSolution(problem=problem,routes=[])
	M=time.time();E=A.route_states
	def N():return bool(A.remaining_pass_pick or A.remaining_pass_drop or A.remaining_parc_pick or A.remaining_parc_drop)
	C={_R:0,_M:0}
	while N():
		C[_R]+=1;H=[A for(A,B)in enumerate(E)if not B[_D]]
		if not H:break
		D=min(H,key=lambda i:E[i][_G]);F=A.possible_actions(D);C[_M]+=len(F)
		if not F:A.apply_return(D);continue
		I,J,K=min(F,key=lambda x:x[2]);A.apply_action(D,I,J,K)
		if G:print(f"[Greedy] Taxi {D} extended route with {I} {J} (inc {K})")
	for(O,P)in enumerate(E):
		if not P[_D]:A.apply_return(O)
	B=A.to_solution();L=time.time()-M;Q={_R:C[_R],_M:C[_M],_L:L}
	if B and not B.is_valid():B=_B
	if G:print('[Greedy] All tasks completed.');print(f"[Greedy] Solution max cost: {B.max_cost if B else _W}");print(f"[Greedy] Time taken: {L:.4f} seconds")
	return B,Q
def iterative_greedy_balanced_solver(problem,partial=_B,iterations=10,destroy_proba=.4,destroy_steps=15,destroy_t=_K,rebuild_proba=.3,rebuild_steps=5,rebuild_t=_K,time_limit=1e1,seed=_B,verbose=_A):
	V='status';M=time_limit;H=verbose;G=seed;F=partial;E=problem
	if F is _B:F=PartialSolution(problem=E,routes=[])
	W=random.Random(G);I=time.time();N=I+M if M is not _B else _B;A,X=greedy_balanced_solver(E,partial=F,verbose=_A)
	if not A:return _B,{_L:time.time()-I,V:'error'}
	C=A.max_cost;O=X[_M];P=0;Q=0;R=0;S='done';J=0
	if H:print(f"[Iterative Greedy] [Iter 0] initial best cost: {C}")
	for T in range(1,iterations+1):
		if N and time.time()>=N:S='timeout';break
		J+=1;K=_B if G is _B else 2*G+T;D,Y,Z=destroy_operator(A,destroy_proba,destroy_steps,seed=K,t=destroy_t);Q+=Z
		for(L,a)in enumerate(Y):
			if not a or len(D.routes[L])<=2:continue
			if W.random()>rebuild_proba:continue
			D,e,b=repair_one_route(D,route_idx=L,steps=rebuild_steps,T=rebuild_t,seed=K+L if K is not _B else _B,verbose=_A);R+=b
		B,c=greedy_balanced_solver(E,partial=D,verbose=_A);O+=c[_M]
		if B and B.is_valid()and B.max_cost<C:
			A=B;C=B.max_cost;P+=1
			if H:print(f"[Iterative Greedy] [Iter {T}] improved best to {C}")
	U=time.time()-I;d={_R:J,'improvements':P,_M:O,'nodes_destroyed':Q,'nodes_rebuilt':R,_L:U,V:S}
	if H:print(f"[Iterative Greedy] Finished after {J} iterations.");print(f"[Iterative Greedy] Best solution max cost: {A.max_cost if A else _W}.");print(f"[Iterative Greedy] Time taken: {U:.4f} seconds.")
	return A,d
def balanced_scorer(parsol,sample_size=15,w_std=.15,seed=_B):
	B=parsol;D=random.Random(seed);E=max(1,sample_size);C=B.route_costs
	if len(C)==1:return B.max_cost
	A=D.choices(C,k=E);F=math.fsum(A)/len(A);G=math.fsum((A-F)**2 for A in A)/len(A);H=math.sqrt(max(.0,G));return B.max_cost+w_std*H
def check_general_action(partial,action):
	A=partial;B,C,D,E=action
	if C==_Y:return A.check_return(B)
	return A.check_action(B,C,D)
def apply_general_action(partial,action):
	A=partial;B,C,D,E=action
	if C==_Y:A.apply_return(B)
	else:A.apply_action(B,C,D,E)
def enumerate_actions_greedily(partial,width=_B,assymetric=_C):
	E=width;A=partial
	if E is _B:E=10**9
	B=A.problem;J=[A for(A,B)in enumerate(A.route_states)if not B[_D]]
	if not J:return[]
	D=sorted(J,key=lambda idx:A.route_states[idx][_G]);K=len(D)
	if assymetric:
		L=set();M=[]
		for C in D:
			N=tuple(A.route_states[C][_J])
			if N in L:continue
			L.add(N);M.append(C)
		D=M
	def U(aggressive):
		if not aggressive:return K
		return min(2 if B.K>=25 else 3 if B.K>=12 else 4 if B.K>=6 else 5,K)
	def V(aggressive):
		if not aggressive:return 10**9
		return min(2 if B.num_nodes>=500 else 4 if B.num_nodes>=200 else 6 if B.num_nodes>=100 else 8 if B.num_nodes>=50 else 12 if B.num_nodes>=25 else 16,E)
	def O(aggressive):
		F=aggressive;B=[];G=D
		if F:G=D[:U(aggressive=_C)]
		for H in G:
			C=A.possible_actions(H)
			if F:C=sorted(C,key=lambda item:item[2])[:V(aggressive=_C)]
			I=[(H,A,B,C)for(A,B,C)in C];B.extend(I);B.sort(key=lambda item:item[3]);B=B[:E]
		return B
	F=O(aggressive=_C)
	if not F:F=O(aggressive=_A)
	W=A.max_cost;G=[];H=[]
	for(C,P,Q,I)in F:
		if A.route_costs[C]+I<=W:G.append((C,P,Q,I))
		else:H.append((C,P,Q,I))
	G.sort(key=lambda item:item[3]);H.sort(key=lambda item:item[3]);R=G+H
	if not R:
		if A.num_actions<B.num_nodes-1:print('[Warning] No feasible actions found before closing depth');raise RuntimeError('Premature routes not covering all nodes.')
		S=[]
		for C in D:
			T=A.route_states[C]
			if T[_F]==0:continue
			if A.check_return(C):X=B.D[T[_F]][0];S.append((C,_Y,0,X))
		return S[:E]
	return R
from typing import Iterator
import bisect
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
			A.size=len(A.arr);A.sum=sum(A.arr);B=0;C=float(_S);D=float(_Z)
			for E in A.arr:B+=E;C=min(C,B);D=max(D,B)
			A.min_pref=C;A.max_pref=D
		def insert(A,idx,entry):A.arr.insert(idx,entry);A.recalc()
		def erase(A,idx):del A.arr[idx];A.recalc()
	def __init__(A,data):A.block_arr=[];A.n_data=0;A.block_prefix=[];A.build(data)
	def build(A,data):
		A.block_arr.clear();A.n_data=len(data);A.block_size=max(0,int(math.sqrt(A.n_data)))+2
		for B in range(0,A.n_data,A.block_size):A.block_arr.append(A.Block(data[B:B+A.block_size]))
		A.n_block=len(A.block_arr);A._rebuild_indexing()
	def _rebuild_indexing(A):
		A.block_prefix=[];B=0
		for C in A.block_arr:A.block_prefix.append(B);B+=C.size
		A.n_data=B
	def _find_block(A,idx):
		B=idx
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
		B=float(_S);A=0;C=0;B=float(_S);A=0
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
		B=float(_Z);A=0;C=0;B=float(_Z);A=0
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
		if B<0 or B>=A.n_data:raise IndexError('Index out of bounds')
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
	P=to_route_idx;O=from_route_idx;N=q_idx_to;M=p_idx_to;I=q_idx_from;H=p_idx_from;C=partial;D=C.routes[O];G=C.routes[P];A=C.problem.D;Y=C.max_cost;E=D[H];F=D[I]
	if H+1==I:B=D[H-1];J=D[I+1];Q=A[B][E]+A[E][F]+A[F][J];R=A[B][J]
	else:B=D[H-1];K=D[H+1];S=D[I-1];J=D[I+1];Q=A[B][E]+A[E][K]+A[S][F]+A[F][J];R=A[B][K]+A[S][J]
	T=C.route_costs[O]-Q+R
	if N==M+1:B=G[M-1];L=G[N-1];U=A[B][L];V=A[B][E]+A[E][F]+A[F][L]
	else:B=G[M-1];K=G[M];W=G[N-2];L=G[N-1];U=A[B][K]+A[W][L];V=A[B][E]+A[E][K]+A[W][F]+A[F][L]
	X=C.route_costs[P]+V-U;Z=max(T,X,*(C.route_costs[A]for A in range(C.problem.K)if A!=O and A!=P));return T,X,Y-Z
def relocate_from_to(partial,from_route_idx,to_route_idx,steps,mode,uplift=1,seed=_B,verbose=_A):
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
				if M==_a:yield(C,D,F,I,L,N,J);return
				else:yield(C,D,F,I,L,N,J)
	def W():
		A=list(V())
		if not A:return
		if M=='stochastic':return P.choice(A)
		elif M=='best':return max(A,key=lambda x:x[6])
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
			if verbose:K,L,N,O,Q,__,P=C;print(f"[Relocate] [{E}->{F}] moved request (P:{K},D:{L}) to ({N},{O}). Decrement={P}")
			if M==_a:break
		return J,I
	a,b=Z();return C,a,b
def relocate_operator(partial,steps=_B,mode=_a,uplift=1,seed=_B,verbose=_A):
	I=steps;G=verbose;D=partial;B=D.problem.K
	if B<2:return D.copy(),[_A]*B,0
	J=I if I is not _B else 10**9;O=random.Random(seed);A=D.copy();K=[_A]*B;C=0
	while C<J:
		L=list(enumerate(A.route_costs));E=max(L,key=lambda x:x[1])[0];P=[A for(A,B)in sorted(L,key=lambda x:x[1])][:max(4,B//2)]
		if len(A.routes[E])<5:break
		M=_A
		for F in P:
			if F==E:continue
			if len(A.routes[F])<2:continue
			Q=J-C;R,S,H=relocate_from_to(A,from_route_idx=E,to_route_idx=F,steps=Q,mode=mode,uplift=uplift,seed=O.randint(10,10**9),verbose=G)
			if H>0:
				A=R;C+=H
				for N in range(B):
					if S[N]:K[N]=_C
				M=_C
				if G:print(f"{H} relocation made from route {E} to route {F}")
				break
		if not M:break
	if G:print();print('[Relocate] Operator completed. ');print(f"Total relocations = {C}; ");print(f"Decrement = {D.max_cost-A.max_cost}; ");print(f"New max cost = {A.max_cost}.");print(_V);print()
	return A,K,C

def _default_defense_policy(partial,seed=_B):A=partial;B,C=B,C=greedy_balanced_solver(A.problem,A,_A);return B
def _default_finalize_policy(partial,seed=_B):
	A=partial;B,E=iterative_greedy_balanced_solver(A.problem,A,iterations=5000,time_limit=5.,seed=seed,verbose=_A)
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
		C=solution;B=C.max_cost;A.count+=1;A.total_cost+=B;A.worst_cost=max(A.worst_cost,B)
		if B<A.best_cost:A.best_cost=B;A.best_solution=C
	def _update_from_swarm(C,swarm):
		for A in swarm.partial_lists:
			if not A.is_complete():continue
			B=A.to_solution()
			if not B:continue
			C._update_from_solution(B)
	def _update_from_list(B,solutions):
		for A in solutions:
			if A is _B:continue
			B._update_from_solution(A)
	def stats(A):B=A.total_cost/A.count if A.count>0 else .0;return{_T:A.best_cost,'worst_cost':A.worst_cost,'avg_cost':B,'count':A.count}
	def opt(A):return A.best_solution
class PheromoneMatrix:
	def __init__(A,problem,sigma,rho,init_cost):A.size=problem.num_nodes;A.sigma=sigma;A.rho=rho;A.tau_0=_K/(rho*init_cost);A.tau_max=2*A.tau_0;A.tau_min=A.tau_0/1e1;A.tau=[[A.tau_0 for B in range(A.size)]for B in range(A.size)]
	def _clamp(A,phe):return min(A.tau_max,max(A.tau_min,phe))
	def get(A,i,j):return A.tau[i][j]
	def set(A,i,j,phe):A.tau[i][j]=A._clamp(phe)
	def update(A,swarm,opt):
		D=opt
		def F(partial):
			B=[]
			for A in partial.routes:
				for C in range(len(A)-1):B.append((A[C],A[C+1]))
			return B
		G=sorted([(A,float(A.max_cost))for A in swarm.partial_lists],key=lambda x:x[1])[:A.sigma-1];E=[[.0 for A in range(A.size)]for B in range(A.size)]
		for(H,(I,J))in enumerate(G,start=1):
			K=(A.sigma-H)/J
			for(B,C)in F(I):
				if 0<=B<A.size and 0<=C<A.size:E[B][C]+=K
		if D is not _B and D.max_cost>0:
			L=A.sigma/D.max_cost;M=PartialSolution.from_solution(D)
			for(B,C)in F(M):
				if 0<=B<A.size and 0<=C<A.size:E[B][C]+=L
		for B in range(A.size):
			for C in range(A.size):N=A.rho*A.tau[B][C]+E[B][C];A.tau[B][C]=A._clamp(N)
class DesirabilityMatrix:
	def __init__(A,problem,phi,chi,gamma,kappa):
		F=problem;A.size=F.num_nodes;A.problem=F;A.phi=phi;A.chi=chi;A.gamma=gamma;A.kappa=kappa;A.eta_dist=[];B=A.problem.D
		for C in range(A.size):
			E=[]
			for D in range(A.size):
				if C==D:E.append(0)
				else:G=max(B[0][C]+B[D][0]-B[C][D],0);H=(1+G)**A.phi;I=(1+B[C][D])**A.chi;J=H/I;E.append(J)
			A.eta_dist.append(E)
	def get(A,i,j,partial,action):
		D,B,E,J=action;F=A.problem.Q[D];G=partial.route_states[D];C=G[_I]
		if B==_P:C+=A.problem.q[E-1]
		if B==_Q:C-=A.problem.q[E-1]
		H=2-int(B==_O);I=(1+A.gamma*(F-C)/F)*A.kappa;return A.eta_dist[i][j]*H*I
class NearestExpansionCache:
	def __init__(C,problem,num_nearest=3):
		A=problem;C.nearest_actions=[]
		for D in range(A.num_nodes):
			if D==0:E=[[0]for A in range(A.K)]
			else:E=[[0,D]]+[[0]for A in range(A.K-1)]
			F=PartialSolution(A,routes=E);B=F.possible_actions(0);B=sorted(B,key=lambda item:item[2])[:num_nearest];C.nearest_actions.append(B)
	def query(J,partial,num_queried):
		A=partial;K=A.max_cost;C=[];D=[]
		for(B,E)in enumerate(A.route_states):
			if E[_D]:continue
			L=E[_F];M=J.nearest_actions[L]
			for N in M:
				F,G,H=N
				if not A.check_action(B,F,G):continue
				I=B,F,G,H
				if A.route_costs[B]+H<=K:C.append(I)
				else:D.append(I)
		return(C+D)[:num_queried]
class Ant:
	class ProbaExpandSampler:
		partial:PartialSolution;cache:'NearestExpansionCache';alpha:float;beta:float;q_prob:float;width:int
		def __init__(A,partial,cache,alpha,beta,q_prob,width):A.partial=partial;A.cache=cache;A.alpha=alpha;A.beta=beta;A.q_prob=q_prob;A.width=width
		def _get_to_node(D,action):
			E,A,B,F=action;C=D.partial.problem
			if A==_O:return C.ppick(B)
			elif A==_U:return C.pdrop(B)
			elif A==_P:return C.parc_pick(B)
			elif A==_Q:return C.parc_drop(B)
			else:return 0
		def _compute_log_proba(A,tau,eta,action):B=action;G=B[0];H=A.partial.route_states[G];E=H[_F];F=A._get_to_node(B);C=tau.get(E,F);D=eta.get(E,F,A.partial,B);C=max(C,1e-300);D=max(D,1e-300);I=A.alpha*math.log(C)+A.beta*math.log(D);return I
		def _collect_actions(B):
			C=B.partial;A=B.width;D=B.cache.query(C,A)
			if D:return D[:A]
			return enumerate_actions_greedily(C,width=A,assymetric=_C)[:A]
		def sample_action(B,tau,eta,rng):
			A=B._collect_actions()
			if not A:return
			C=[]
			for F in A:G=B._compute_log_proba(tau,eta,F);C.append(G)
			H=max(C);E=[math.exp(A-H)for A in C];I=sum(E);J=[A/I for A in E];D:0
			if rng.random()<B.q_prob:D=min(range(len(A)),key=lambda i:A[i][3])
			else:D=sample_from_weight(rng,J)
			return A[D]
	def __init__(A,partial,cache,tau,eta,alpha,beta,q_prob,width,rng):F=width;E=q_prob;D=alpha;C=cache;B=partial;A.problem=B.problem;A.partial=B;A.cache=C;A.tau=tau;A.eta=eta;A.alpha=D;A.beta=beta;A.q_prob=E;A.width=F;A.rng=rng;A.sampler=Ant.ProbaExpandSampler(partial=A.partial,cache=C,alpha=D,beta=beta,q_prob=E,width=F)
	def expand(A):
		if A.partial.is_complete():return _A
		B=A.sampler.sample_action(A.tau,A.eta,A.rng)
		if not B:return _A
		apply_general_action(A.partial,B);return _C
class AntPopulation:
	def __init__(A,initial_swarm,cache,tau,eta,lfunc,alpha,beta,q_prob,width,iterations,time_limit,seed,verbose):
		D=cache;C=initial_swarm;B=seed;A.swarm=C.copy();A.completed=[A.is_complete()for A in A.swarm.partial_lists];A.cache=D;A.tau=tau;A.eta=eta;A.lfunc=lfunc;A.iterations=iterations;A.time_limit=time_limit;A.seed=B;A.verbose=verbose;A.ants=[]
		for(E,F)in enumerate(A.swarm.partial_lists):G=Ant(partial=F,cache=D,tau=A.tau,eta=A.eta,alpha=alpha,beta=beta,q_prob=q_prob,width=width,rng=random.Random(hash(B+100*E)if B else _B));A.ants.append(G)
		A.num_ants=len(A.ants);A.max_actions=C.problem.num_actions;A.start_time=time.time();A.end_time=A.start_time+A.time_limit;A.tle=lambda:time.time()>A.end_time
	def expand(A):
		D=_A
		for(B,C)in enumerate(A.ants):
			if A.completed[B]:continue
			if C.expand():D=_C
			elif A.verbose:print(f"[ACO] [Depth {C.partial.num_actions}] [Warning] Ant {B+1} cannot expand, further diagnosis needed.")
			A.completed[B]=C.partial.is_complete()
		return D
	def update(A):A.lfunc.update(source=A.swarm);A.tau.update(swarm=A.swarm,opt=A.lfunc.opt())
	def run(A):
		G=[A.max_actions//5,A.max_actions//2,A.max_actions*9//10]
		for B in range(A.iterations):
			if A.tle():
				if A.verbose:print('[ACO] Time limit reached, skipping iteration.')
				return A.swarm
			if A.verbose:
				if B in G or B%100==0:C=[A.max_cost for A in A.swarm.partial_lists];D=[A.num_actions for A in A.swarm.partial_lists];print(f"[ACO] [Iteration {B}] Partial cost range: {min(C):.3f} - {max(C):.3f}, Depth range: {min(D)} - {max(D)}, Time_elapsed={time.time()-A.start_time:.2f}s.")
			if not A.expand():
				if A.verbose:print('[ACO] All ants have completed their solutions.')
				break
			A.update()
		if A.verbose:H=sum(1 for A in A.swarm.partial_lists if A.is_complete());E=A.swarm.opt();F=A.lfunc.opt();print(f"[ACO] Finished all iterations.\nComplete solutions found: {H}/{A.num_ants}.\nRun best cost: {E.max_cost if E else _W}, Opt cost: {F.max_cost if F else _W}.")
		return A.swarm
class SwarmTracker:
	def __init__(A,initial_swarm,defense_policy,finalize_policy,seed=_B):C=defense_policy;B=initial_swarm;A.seed=seed;A.frontier_swarm=[A.copy()for A in B.partial_lists];A.num_partials=B.num_partials;A.frontier_potential=[C(A)for A in A.frontier_swarm];A.frontier_potential_costs=[A.max_cost if A is not _B else 10**18 for A in A.frontier_potential];(A.finals):0;A.is_finalized=_A;A.defense_policy=C;A.finalize_policy=finalize_policy
	def update(A,source):
		for(B,D)in enumerate(source.partial_lists):
			C=A.defense_policy(D,seed=A.seed+10*B if A.seed else _B)
			if not C:continue
			E=C.max_cost
			if E<A.frontier_potential_costs[B]:A.frontier_swarm[B]=D.copy();A.frontier_potential[B]=C;A.frontier_potential_costs[B]=E
		return A.frontier_potential
	def finalize(A,cutoff,time_limit):
		D=time_limit;C=cutoff
		if D is _B:D=float(_S)
		J=time.time();K=J+D;E=sorted(zip(A.frontier_swarm,A.frontier_potential,A.frontier_potential_costs),key=lambda x:x[2]);L=E[:C]if C else E;B=[];F=[]
		for(G,(M,O,P))in enumerate(L):
			if time.time()>=K:
				for(N,H,N)in E[G:]:
					if H is not _B:F.append(H)
				break
			I=A.finalize_policy(M,seed=A.seed+20*G if A.seed else _B)
			if I:B.append(I)
		B.extend(F);B.sort(key=lambda s:s.max_cost);B=B[:C]if C else B;A.is_finalized=_C;A.finals=B;return B
	def top(A,k,cutoff=_B,time_limit=_B):
		B=cutoff
		if B is _B:B=k
		if not A.is_finalized:A.finalize(B,time_limit)
		return A.finals[:k]
	def opt(A,cutoff=_B,time_limit=_B):
		if not A.is_finalized:A.finalize(cutoff,time_limit)
		return A.finals[0]
def _run_aco(problem,swarm,runs,iterations,width,cutoff,q_prob,alpha,beta,phi,chi,gamma,kappa,sigma,rho,defense_policy,finalize_policy,seed,time_limit,verbose):
	L=swarm;I=time_limit;H=cutoff;G=iterations;D=problem;B=runs;A=verbose;J=time.time();M=J+I
	if G is _B:G=D.num_actions
	if A:print('[ACO] [Init] Estimating costs from initial greedy solver...')
	N,Z=greedy_balanced_solver(D,_B,_A);O=N.max_cost
	if A:print(f"[ACO] [Init] Greedy solution cost: {O:.3f}")
	if A:print('[ACO] [Init] Initializing nearest expansion cache...')
	R=NearestExpansionCache(D,num_nearest=20)
	if A:print('[ACO] [Init] Initializing matrices...')
	S=PheromoneMatrix(D,sigma=sigma,rho=rho,init_cost=O);T=DesirabilityMatrix(D,phi,chi,gamma,kappa)
	if A:print('[ACO] [Init] Initializing trackers...')
	K=SolutionTracker();K.update(N);E=SwarmTracker(initial_swarm=L,defense_policy=defense_policy,finalize_policy=finalize_policy);P=B
	for C in range(B):
		if time.time()-J>=.8*I:
			P=C
			if A:print(f"[ACO] Time limit approaching, stopping at run {C+1}/{B}.")
			break
		if A:print(f"[ACO] [Run {C+1}/{B}] Starting the population run...")
		U=AntPopulation(initial_swarm=L,cache=R,tau=S,eta=T,lfunc=K,alpha=alpha,beta=beta,q_prob=q_prob,width=width,iterations=G,time_limit=I,seed=hash(seed+10*C)if seed else _B,verbose=A)
		if A:print(f"[ACO] [Run {C+1}/{B}] Running the ant population")
		V=U.run()
		if A:print(f"[ACO] [Run {C+1}/{B}] Updating swarm tracker")
		W=E.update(V);K.update(W)
		if A:print()
	if A:print(f"[ACO] Finalizing top {H} partial into solutions...")
	E.finalize(H,max(0,M-time.time()));X=time.time()-J;Q=E.opt(cutoff=H,time_limit=max(0,M-time.time()));Y=Q.max_cost if Q else float(_S);F={_X:P,_L:X,_T:Y,_b:E.num_partials}
	if A:print(f"[ACO] The run finished. Runs_completed={F[_X]}, Best_cost={F[_T]:.3f}, Time={F[_L]:.3f}s.")
	return E,F
def aco_solver(problem,initial_swarm=_B,cutoff=5,num_ants=10,runs=10,iterations=_B,width=10,q_prob=.75,alpha=1.2,beta=1.4,phi=.5,chi=1.5,gamma=.4,kappa=2.,sigma=10,rho=.55,defense_policy=_default_defense_policy,finalize_policy=_default_finalize_policy,seed=_B,time_limit=3e1,verbose=_A):
	E=verbose;D=problem;B=initial_swarm
	if B is _B:G=[PartialSolution(problem=D,routes=[])for A in range(num_ants)];B=PartialSolutionSwarm(solutions=G)
	H,A=_run_aco(problem=D,swarm=B,runs=runs,iterations=iterations,width=width,q_prob=q_prob,alpha=alpha,beta=beta,phi=phi,chi=chi,gamma=gamma,kappa=kappa,sigma=sigma,rho=rho,defense_policy=defense_policy,finalize_policy=finalize_policy,cutoff=cutoff,seed=seed,time_limit=time_limit,verbose=E);C=H.opt();F={_X:A[_X],_L:A[_L],_T:A[_T],_b:A[_b]}
	if E:
		print();print('[ACO] Solver complete.')
		if C is not _B:print(f"[ACO] Best solution cost: {C.max_cost}")
		else:print('[ACO] No valid solution found.')
		print(f"[ACO] Total time: {F[_L]:.3f}s");print(_V);print()
	return C,F
def read_instance():
	A,B,D=map(int,sys.stdin.readline().strip().split());E=list(map(int,sys.stdin.readline().split()));F=list(map(int,sys.stdin.readline().split()));C=[[0]*(2*A+2*B+1)for C in range(2*A+2*B+1)]
	for G in range(2*A+2*B+1):H=sys.stdin.readline().strip();C[G]=list(map(int,H.split()))
	return ShareARideProblem(A,B,D,E,F,C)
def main(verbose=_A):B=verbose;A=read_instance();C,D=aco_solver(A,cutoff=10,num_ants=500 if A.num_nodes<=100 else 150 if A.num_nodes<=250 else 50 if A.num_nodes<=500 else 25 if A.num_nodes<=1000 else 10,runs=100 if A.num_nodes<=100 else 75 if A.num_nodes<=250 else 50 if A.num_nodes<=500 else 20 if A.num_nodes<=1000 else 10,width=10 if A.num_nodes<=100 else 8 if A.num_nodes<=250 else 6 if A.num_nodes<=500 else 4 if A.num_nodes<=1000 else 2,seed=42,time_limit=25e1,verbose=B);C.stdin_print(verbose=B)
if __name__=='__main__':main(verbose=_A)