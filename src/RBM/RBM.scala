package RBM

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions.Gaussian
import breeze.numerics.sigmoid
import breeze.linalg.InjectNumericOps
object RBM {
    val n_visible=10
    val n_hidden=5
    val alpha=0.02
    val momentum=0.1
    
    var W=DenseMatrix.rand(n_visible,n_hidden, new Gaussian(0,0.01))
    var b=DenseVector.rand(n_visible, new Gaussian(0,0.01))
    var c=DenseVector.zeros[Double](n_hidden)
    var vW=DenseMatrix.zeros[Double](n_visible,n_hidden)
    var vb=DenseVector.zeros[Double](n_visible)
    var vc=DenseVector.zeros[Double](n_hidden)
    
    def tune(vis:String)= {
      val v1=new DenseVector(vis.split(',').map(x=>x.toDouble))
      //println("v1:"+v1.toString)
      val h1=sigmrnd(c  +  W.t*v1)
      //println("h1:"+h1.toString)
      val v2=sigmrnd(b  +  W*h1)
      //println("v2"+v2.toString)
      val h2=sigmrnd(c  +  W.t*v2)
      //println("h2"+h2.toString)
      /*h.t*v is a scala while h*v.t is a matrix; DenseVector is a colunm vector*/
      val c1=h1*v1.t
      val c2=h2*v2.t
      //DenseVector.fill(vW.length){momentum}.t*vW + DenseVector.fill(vW.length){alpha*(c1-c2)}
      //delta W, delta b, delta c, error
      (alpha*(c1-c2),alpha*(v1-v2),alpha*(h1-h2),(v1-v2).t*(v1-v2))
    }
    def sigmrnd(vec:DenseVector[Double])={
      val a=sigmoid(vec) :> DenseVector.rand(vec.length)
      val a1=a.map { f => if(f==true) 1.0 else 0.0 }
      a1
    }
}
