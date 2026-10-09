"""Check the smoothing diagnostic against direct categorical enumeration."""
import math
import numpy as np

from scripts.evaluate_lag_disagreement import multinomial_forecast
from scripts.evaluate_lag_reference_sensitivity import target_conditional_multinomial


def test_conditional_at_target_variance_matches_enumeration():
    reference=np.array([[0,1,0,2],[2,0,1,2]],dtype=np.int32)
    evaluation=np.array([[0,1,2,1]],dtype=np.int32)
    mask=np.array([[False,True,False,True]])
    prior=np.array([.2,.3,.5])
    alpha=2.
    gram,cross=np.array([[2.]]),np.array([1.])
    for n in (2,4,8):
        actual=target_conditional_multinomial(reference,evaluation,mask,prior,(1,),gram,cross,n=n,smoothing=alpha,ridge=0.)[0]
        per_position=[]
        for context in (0,2):
            counts=np.zeros(3)
            for row in reference:
                for i in range(1,len(row)):
                    if row[i-1]==context:
                        counts[row[i]]+=1
            scaled=counts*n/len(reference)
            N=int(scaled.sum())
            theta=(scaled+alpha*prior)/(N+alpha)
            mean,second=np.zeros(3),0.
            for a in range(N+1):
                for b in range(N-a+1):
                    y=np.array([a,b,N-a-b])
                    mass=math.factorial(N)/math.prod(math.factorial(int(v)) for v in y)*np.prod(theta**y)
                    estimate=(y+alpha*prior)/(N+alpha)
                    mean+=mass*estimate
                    second+=mass*float(estimate@estimate)
            per_position.append(2*(second-float(mean@mean))*.5**2)
        np.testing.assert_allclose(actual,np.mean(per_position),rtol=1e-11,atol=1e-13)
        if n==len(reference):
            original=multinomial_forecast(reference,evaluation,mask,prior,(1,),gram,cross,n=n,smoothing=alpha,ridge=0.)
            np.testing.assert_allclose(actual,original[0],rtol=0,atol=1e-14)
