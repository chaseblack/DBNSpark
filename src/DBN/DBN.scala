package DBN

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
//import org.apache.spark.internal.Logging
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.{Matrix => BM,CSCMatrix => BSM,DenseMatrix => BDM, Vector => BV,DenseVector => BDV,SparseVector => BSV,axpy => brzAxpy,svd => brzSvd}
import breeze.numerics.{exp => Bexp,tanh => Btanh}
import breeze.stats.distributions.Gaussian
import scala.collection.mutable.ArrayBuffer
import java.util.Random
import scala.math._

case class DBNParams(W: BDM[Double], vW: BDM[Double],b: BDM[Double], vb: BDM[Double], c: BDM[Double], vc: BDM[Double]) extends Serializable
case class GaussianDBNParams(W: BDM[Double], vW: BDM[Double],b: BDM[Double], vb: BDM[Double], c: BDM[Double], vc: BDM[Double], deviat: BDM[Double], v_inv_deviat: BDM[Double]) extends Serializable
case class DBNHyperParams(size: Array[Int], layer: Int, momentum: Double, alpha: Double) extends Serializable

class DBN(private var size: Array[Int], private var layer: Int, private var momentum: Double, private var alpha: Double) extends Serializable {
  //var size=Array(5, 10, 10): network arch.  var layer=3 <=> numel(nn.size).   var momentum=0.0;    var alpha=1.0
  def this() = this(DBN.Size, 3, 0.0, 0.2)// 
  def setSize(size: Array[Int]): this.type = {this.size = size;this}/** 设置神经网络结构. Default: [10, 5, 1]. */
  def setLayer(layer: Int): this.type = {this.layer = layer;this}/** 设置神经网络层数据. Default: 3. */
  def setMomentum(momentum: Double): this.type = {this.momentum = momentum;this}/** 设置Momentum. Default: 0.0. */
  def setAlpha(alpha: Double): this.type = {this.alpha = alpha;this}/**set alpha, the default is 0.2*/
  
  def DBNtrain(train_d: RDD[(BDM[Double]/*vector*/, BDM[Double]/*vector*/)],  opts: Array[Double]) : DBNModel = {
    val sc = train_d.sparkContext
    val dbnconfig = DBNHyperParams(size, layer, momentum, alpha)
    // init params
    var dbn_deviat=DBN.InitVar(size)
    var dbn_v_inv_deviat=DBN.InitvVar(size)
    var dbn_W = DBN.InitialW(size)
    var dbn_vW = DBN.InitialvW(size)
    var dbn_b = DBN.Initialb(size)
    var dbn_vb = DBN.Initialvb(size)
    var dbn_c = DBN.Initialc(size)
    var dbn_vc = DBN.Initialvc(size)
    
    printf("Train the first RBM.........")
    val param0 = new GaussianDBNParams(dbn_W(0), dbn_vW(0), dbn_b(0), dbn_vb(0), dbn_c(0), dbn_vc(0),dbn_deviat,dbn_v_inv_deviat)
    val param1 = GaussianRBMtrain(train_d, opts, dbnconfig, param0)
    dbn_W(0) = param1.W
    dbn_vW(0) = param1.vW
    dbn_b(0) = param1.b
    dbn_vb(0) = param1.vb
    dbn_c(0) = param1.c
    dbn_vc(0) = param1.vc
    
    printf("Train the rest RBMs.........")
    for (i <- 2 to dbnconfig.layer - 1) {
      printf("Train %d th RBM...", i)
      val tmp_bc_w = sc.broadcast(dbn_W(i - 2))
      val tmp_bc_c = sc.broadcast(dbn_c(i - 2))
      val train_d2 = train_d.map { f =>// 前向计算x x = sigm(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');
        val label = f._1
        val x = f._2
        val x2 = DBN.sigm(x * tmp_bc_w.value.t + tmp_bc_c.value.t)
        (label, x2)
      }
      val param_i = new DBNParams(dbn_W(i - 1), dbn_vW(i - 1), dbn_b(i - 1), dbn_vb(i - 1), dbn_c(i - 1), dbn_vc(i - 1))
      val weight2 = RBMtrain(train_d2, opts, dbnconfig, param_i)
      dbn_W(i - 1) = weight2.W
      dbn_vW(i - 1) = weight2.vW
      dbn_b(i - 1) = weight2.b
      dbn_vb(i - 1) = weight2.vb
      dbn_c(i - 1) = weight2.c
      dbn_vc(i - 1) = weight2.vc
    }
    new DBNModel(dbnconfig, dbn_W, dbn_b, dbn_c)
  }
  
  
  def GaussianRBMtrain(train_t: RDD[(BDM[Double], BDM[Double])], opts: Array[Double], hyperparams: DBNHyperParams, weight: GaussianDBNParams): GaussianDBNParams = {
    val sc = train_t.sparkContext
    var rbm_deviat=weight.deviat
    var rbm_v_inv_deviat=weight.v_inv_deviat
    var rbm_W = weight.W
    var rbm_vW = weight.vW
    var rbm_b = weight.b
    var rbm_vb = weight.vb
    var rbm_c = weight.c
    var rbm_vc = weight.vc
    val broadcast_config = sc.broadcast(hyperparams)
    val m = train_t.count// number of train samples
    val batchsize = opts(0).toInt
    val numepochs = opts(1).toInt
    val numbatches = (m / batchsize).toInt
    
    for (i <- 1 to numepochs) {
      val splitW2 = Array.fill(numbatches)(1.0 / numbatches)//if numbatches=10, then Array(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
      var err = 0.0
      for (l <- 1 to numbatches) {//not paralleled, a paralleled one should be compute each batch simultaneously
        val broadcast_rbm_W = sc.broadcast(rbm_W)//num.of.layer_latter * num.of.layer_former 
        val broadcast_rbm_vW = sc.broadcast(rbm_vW)//num.of.layer_latter * num.of.layer_former 
        val broadcast_rbm_b = sc.broadcast(rbm_b)//col vector
        val broadcastc_rbm_vb = sc.broadcast(rbm_vb)//col vector
        val broadcast_rbm_c = sc.broadcast(rbm_c)//col vector
        val broadcast_rbm_vc = sc.broadcast(rbm_vc)//col vector
        val broadcast_rbm_deviat = sc.broadcast(rbm_deviat)//col vector
        val broadcast_rbm_v_inv_deviat = sc.broadcast(rbm_v_inv_deviat)//col vector
        //val broadcast_rbm_deviat=broadcast_rbm_deviat.value.map { x => sqrt(x) }
        
        val train_split2 = train_t.randomSplit(splitW2, System.nanoTime())/**train_split2 has numbatches batches*/
        val batch_xy1 = train_split2(l - 1)//batch_xy1 is RDD[row vector, row vector] mathematically eventhough it is shown as RDD[BDM,BDM]
        val batch_vh1 = batch_xy1.map { f =>// feed forward
          val label = f._1//label is row vector, even though it is shown as BDM
          val v0 = f._2//v1 is row vector, even though it is shown as BDM
          //println("v0 row="+v0.rows+" v0 col="+v0.cols)
          //println("rbm_deviat row="+broadcast_rbm_deviat.value.t.rows+" rbm_deviat col="+broadcast_rbm_deviat.value.t.cols)
          //println("W row="+broadcast_rbm_W.value.t.rows+" W col="+broadcast_rbm_W.value.t.cols)
          //println("rbm_c row="+broadcast_rbm_c.value.t.rows+" rbm_c cols"+broadcast_rbm_c.value.t.cols)
          val h0 = DBN.sigm(((v0:/broadcast_rbm_deviat.value.t) * broadcast_rbm_W.value.t + broadcast_rbm_c.value.t))//row vector
          var h_states=DBN.samplestates(h0)//row vector
          var v1 = h_states * broadcast_rbm_W.value:*broadcast_rbm_deviat.value.t + broadcast_rbm_b.value.t + BDM.rand(1,broadcast_rbm_deviat.value.rows):*broadcast_rbm_deviat.value.t//v1 row vector
          var h1 = DBN.sigm((v1:/broadcast_rbm_deviat.value.t) * broadcast_rbm_W.value.t + broadcast_rbm_c.value.t)//row vector
          for(i<-2 to opts(2).toInt){//k_CD
            h_states=DBN.samplestates(h0)
            v1 = h_states * broadcast_rbm_W.value:*broadcast_rbm_deviat.value.t + broadcast_rbm_b.value.t+BDM.rand(1,broadcast_rbm_deviat.value.rows):*broadcast_rbm_deviat.value.t//v1 row vector
            h1 = DBN.sigm((v1:/broadcast_rbm_deviat.value.t) * broadcast_rbm_W.value.t + broadcast_rbm_c.value.t)//row vector
          }
          val c1 = h0.t * (v0:/broadcast_rbm_deviat.value.t)//delta matrix contributed by ONE SINGLE training sample
          val c2 = h1.t * (v1:/broadcast_rbm_deviat.value.t)//delta matrix contributed by ONE SINGLE training sample
          val d1 =  v0:*(broadcast_rbm_b.value.t*2.toDouble-v0):/broadcast_rbm_deviat.value.t  +  v0:*(h0 * broadcast_rbm_W.value)//row vec
          val d2 =  v1:*(broadcast_rbm_b.value.t*2.toDouble-v1):/broadcast_rbm_deviat.value.t  +  v1:*(h1 * broadcast_rbm_W.value)//BDM.fill[Double](1, v0.cols)(2)
          (label, v0, h0, v1, h1, c1, c2, d1, d2)
        }
        
        // UPDATE W: rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (c1 - c2) / opts.batchsize;
        val vw1 = batch_vh1.map {/*vw1 is RDD[Matrix]*/case (label, v1, h1, v2, h2, c1, c2, d1, d2) =>(c1 - c2)}
        val initw = BDM.zeros[Double](broadcast_rbm_W.value.rows, broadcast_rbm_W.value.cols)
        /**c is type of U=(matrix, count), c._1 is matrix, c._2 is current count; v is type of T, i.e, delta matrix contributed by ONE SINGLE sample; 0L is zeroVal,i.e, U*/
        val (vw2, countw2) = vw1.treeAggregate((initw, 0L))(seqOp = (c, v) => (c._1 + v, c._2 + 1),combOp = (c1, c2) => (c1._1 + c2._1, c1._2 + c2._2))
        val vw3 = vw2 / countw2.toDouble
        rbm_vW = broadcast_config.value.momentum * broadcast_rbm_vW.value + broadcast_config.value.alpha * vw3
        rbm_W = broadcast_rbm_W.value + rbm_vW
        //UPDATE  b: rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize; 
        val vb1 = batch_vh1.map {case (lable, v1, h1, v2, h2, c1, c2, d1, d2) =>(v1 - v2):/broadcast_rbm_deviat.value.map{x => x*x}.t}
        val initb = BDM.zeros[Double](broadcastc_rbm_vb.value.cols, broadcastc_rbm_vb.value.rows)
        val (vb2, countb2) = vb1.treeAggregate((initb, 0L))(seqOp = (c, v) => (c._1 + v, c._2 + 1),combOp = (c1, c2) =>(c1._1 + c2._1, c1._2 + c2._2))
        val vb3 = vb2 / countb2.toDouble
        rbm_vb = broadcast_config.value.momentum * broadcastc_rbm_vb.value + broadcast_config.value.alpha * vb3.t
        rbm_b = broadcast_rbm_b.value + rbm_vb
        //UPDATE c: rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize;
        val vc1 = batch_vh1.map {case (lable, v1, h1, v2, h2, c1, c2, d1, d2) =>(h1 - h2)}
        val initc = BDM.zeros[Double](broadcast_rbm_vc.value.cols, broadcast_rbm_vc.value.rows)
        val (vc2, countc2) = vc1.treeAggregate((initc, 0L))(seqOp = (c, v) =>(c._1 + v, c._2 + 1),combOp = (c1, c2) =>(c1._1 + c2._1, c1._2 + c2._2))
        val vc3 = vc2 / countc2.toDouble
        rbm_vc = broadcast_config.value.momentum * broadcast_rbm_vc.value + broadcast_config.value.alpha * vc3.t
        rbm_c = broadcast_rbm_c.value + rbm_vc
        //UPDATE var
        val v_inv_deviat1 = batch_vh1.map {case (lable, v1, h1, v2, h2, c1, c2, d1, d2) =>(d1 - d2)}
        val init_v_inv_deviat1 = BDM.zeros[Double](broadcast_rbm_deviat.value.cols, broadcast_rbm_deviat.value.rows)
        val (v_inv_deviat2, count) = vc1.treeAggregate((init_v_inv_deviat1, 0L))(seqOp = (c, v) =>(c._1 + v, c._2 + 1),combOp = (c1, c2) =>(c1._1 + c2._1, c1._2 + c2._2))
        val v_inv_deviat3=v_inv_deviat2 / count.toDouble//row vec
        rbm_v_inv_deviat=broadcast_config.value.momentum*rbm_v_inv_deviat+v_inv_deviat3.t
        val inv_deviat=1.toDouble/broadcast_rbm_deviat.value + rbm_v_inv_deviat
        rbm_deviat=1.toDouble/inv_deviat
        rbm_deviat=rbm_deviat.map { x => if(x>0.005) x else 0.005 }

        val dbne1 = batch_vh1.map {// compute error
          case (lable, v1, h1, v2, h2, c1, c2, d1, d2) =>
            (v1 - v2)
        }
        val (dbne2, counte) = dbne1.treeAggregate((0.0, 0L))(
          seqOp = (c, v) => {
            // c: (e, count), v: (m)
            val e1 = c._1
            val e2 = (v :* v).sum
            val esum = e1 + e2
            (esum, c._2 + 1)
          },
          combOp = (c1, c2) => {
            // c: (e, count)
            val e1 = c1._1
            val e2 = c2._1
            val esum = e1 + e2
            (esum, c1._2 + c2._2)
          })
        val dbne = dbne2 / counte.toDouble
        err += dbne
      }
      printf("epoch: numepochs = %d ; Average reconstruction error is: %f.\n", i, err / numbatches.toDouble)
    }
    new GaussianDBNParams(rbm_W, rbm_vW, rbm_b, rbm_vb, rbm_c, rbm_vc,rbm_deviat,rbm_v_inv_deviat)
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  /***********************ATTENTION**************************************
      * Mathematically, each sample in the training set should be a DenseVector. 
      * However, in this code each sample is wrapped in a DenseMatrix, train_t. 
      * Why? I personally think this is because multiplication between a vector 
      * and a matrix in Breeze is tricky from my prior experience on Breeze.
      * 
      * BDM(1,x.length,x) is actually a vector, mathematically. the row number is 1, 
      * the col number is x.length, and the vector data are from x itself. 
      * Therefore I denote those BDMs that are actually vectors.
   **/
  def RBMtrain(train_t: RDD[(BDM[Double]/*row vector*/, BDM[Double]/*row vector*/)], opts: Array[Double], hyperparams: DBNHyperParams, weight: DBNParams): DBNParams = {
    val sc = train_t.sparkContext
    var StartTime = System.currentTimeMillis()
    var EndTime = System.currentTimeMillis()
    var rbm_W = weight.W
    var rbm_vW = weight.vW
    var rbm_b = weight.b
    var rbm_vb = weight.vb
    var rbm_c = weight.c
    var rbm_vc = weight.vc
    val broadcast_config = sc.broadcast(hyperparams)
    val m = train_t.count// number of train samples
    val batchsize = opts(0).toInt
    val numepochs = opts(1).toInt
    val numbatches = (m / batchsize).toInt
    
    for (i <- 1 to numepochs) {
      StartTime = System.currentTimeMillis()
      val splitW2 = Array.fill(numbatches)(1.0 / numbatches)//if numbatches=10, then Array(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
      var err = 0.0
      for (l <- 1 to numbatches) {
        val broadcast_rbm_W = sc.broadcast(rbm_W)//num.of.layer_latter * num.of.layer_former 
        val broadcast_rbm_vW = sc.broadcast(rbm_vW)//num.of.layer_latter * num.of.layer_former 
        val broadcast_rbm_b = sc.broadcast(rbm_b)//col vector
        val broadcastc_rbm_vb = sc.broadcast(rbm_vb)//col vector
        val broadcast_rbm_c = sc.broadcast(rbm_c)//col vector
        val broadcast_rbm_vc = sc.broadcast(rbm_vc)//col vector
        val train_split2 = train_t.randomSplit(splitW2, System.nanoTime())/**train_split2 has numbatches batches*/
        val batch_xy1 = train_split2(l - 1)//batch_xy1 is RDD[row vector, row vector] mathematically eventhough it is shown as RDD[BDM,BDM]
        val batch_vh1 = batch_xy1.map { f =>// feed forward
          val label = f._1//label is row vector, even though it is shown as BDM
          val v0 = f._2//v1 is row vector, even though it is shown as BDM
          val h0 = DBN.sigm((v0 * broadcast_rbm_W.value.t + broadcast_rbm_c.value.t))//row vector
          val h_states=DBN.samplestates(h0)
          val v1 = DBN.sigm((h_states * broadcast_rbm_W.value + broadcast_rbm_b.value.t))//row vector
          val h1 = DBN.sigm(v1 * broadcast_rbm_W.value.t + broadcast_rbm_c.value.t)//row vector
          val c1 = h0.t * v0//delta matrix contributed by ONE SINGLE training sample
          val c2 = h1.t * v1//delta matrix contributed by ONE SINGLE training sample
          (label, v0, h0, v1, h1, c1, c2)
        }
        
        /** update W: rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (c1 - c2) / opts.batchsize;*/
        val vw1 = batch_vh1.map {/*vw1 is RDD[Matrix]*/case (label, v1, h1, v2, h2, c1, c2) =>(c1 - c2)}
        val initw = BDM.zeros[Double](broadcast_rbm_W.value.rows, broadcast_rbm_W.value.cols)
        /**
         * treeAggregate(U)(seqOp(U,T):U,  combOp(U,U):U):U
         * This method takes 2 inputs: U and (seqOP, combOp), and outputs U
         * T is immutable and is one element in RDD; and U can be any structure defined by yourself.
         * here T is element in vw1. here U is (delta matrix contributed by N samples, N)
         * */
        val (vw2, countw2) = vw1.treeAggregate(  (initw, 0L)/*zeroVal,i.e, U*/  )( 
            seqOp = (c/*U=(matrix, count)*/, v/*T, i.e, delta matrix contributed by ONE SINGLE sample*/) => (c._1 + v, c._2 + 1)    
            ,    
            combOp = (c1, c2) => (c1._1 + c2._1, c1._2 + c2._2)  
            )
        val vw3 = vw2 / countw2.toDouble
        rbm_vW = broadcast_config.value.momentum * broadcast_rbm_vW.value + broadcast_config.value.alpha * vw3
        
        /**update  b: rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize; */
        val vb1 = batch_vh1.map {case (lable, v1, h1, v2, h2, c1, c2) =>(v1 - v2)}
        val initb = BDM.zeros[Double](broadcastc_rbm_vb.value.cols, broadcastc_rbm_vb.value.rows)
        val (vb2, countb2) = vb1.treeAggregate((initb, 0L))(
          seqOp = (c, v) => (c._1 + v, c._2 + 1)
          ,
          combOp = (c1, c2) =>(c1._1 + c2._1, c1._2 + c2._2)
          )
        val vb3 = vb2 / countb2.toDouble
        rbm_vb = broadcast_config.value.momentum * broadcastc_rbm_vb.value + broadcast_config.value.alpha * vb3.t
        
        /**update c: rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize;*/
        val vc1 = batch_vh1.map {case (lable, v1, h1, v2, h2, c1, c2) =>(h1 - h2)}
        val initc = BDM.zeros[Double](broadcast_rbm_vc.value.cols, broadcast_rbm_vc.value.rows)
        val (vc2, countc2) = vc1.treeAggregate((initc, 0L))(
          seqOp = (c, v) =>(c._1 + v, c._2 + 1)
          ,
          combOp = (c1, c2) =>(c1._1 + c2._1, c1._2 + c2._2)
          )
        val vc3 = vc2 / countc2.toDouble
        rbm_vc = broadcast_config.value.momentum * broadcast_rbm_vc.value + broadcast_config.value.alpha * vc3.t
        
        rbm_W = broadcast_rbm_W.value + rbm_vW
        rbm_b = broadcast_rbm_b.value + rbm_vb
        rbm_c = broadcast_rbm_c.value + rbm_vc
        
        // compute error
        val dbne1 = batch_vh1.map {
          case (lable, v1, h1, v2, h2, c1, c2) =>
            (v1 - v2)
        }
        val (dbne2, counte) = dbne1.treeAggregate((0.0, 0L))(
          seqOp = (c, v) => {
            // c: (e, count), v: (m)
            val e1 = c._1
            val e2 = (v :* v).sum
            val esum = e1 + e2
            (esum, c._2 + 1)
          },
          combOp = (c1, c2) => {
            // c: (e, count)
            val e1 = c1._1
            val e2 = c2._1
            val esum = e1 + e2
            (esum, c1._2 + c2._2)
          })
        val dbne = dbne2 / counte.toDouble
        err += dbne
      }
      EndTime = System.currentTimeMillis()
      
      printf("epoch: numepochs = %d , Took = %d seconds; Average reconstruction error is: %f.\n", i, scala.math.ceil((EndTime - StartTime).toDouble / 1000).toLong, err / numbatches.toDouble)
    }
    new DBNParams(rbm_W, rbm_vW, rbm_b, rbm_vb, rbm_c, rbm_vc)
  }
  


}






object DBN extends Serializable {
  // Initialization mode names
  val Activation_Function = "sigm"
  val Output = "linear"
  val Size = Array(10, 5, 1)

  def InitialW(size: Array[Int]): Array[BDM[Double]] = {
    // 初始化权重参数
    // weights and weight momentum
    // dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));
    val n = size.length
    val rbm_W = ArrayBuffer[BDM[Double]]()
    for (i <- 1 to n - 1) {
      //val d1 = BDM.zeros[Double](size(i), size(i - 1))
      val d1=BDM.rand(size(i), size(i - 1), new Gaussian(0,0.01))
      rbm_W += d1
    }
    rbm_W.toArray
  }

  /**
   * 初始化权重vW
   * 初始化为0
   */
  def InitialvW(size: Array[Int]): Array[BDM[Double]] = {
    // 初始化权重参数
    // weights and weight momentum
    // dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));
    val n = size.length
    val rbm_vW = ArrayBuffer[BDM[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = BDM.zeros[Double](size(i), size(i - 1))
      rbm_vW += d1
    }
    rbm_vW.toArray
  }

  /**
   * 初始化偏置向量b
   * 初始化为0
   */
  def Initialb(size: Array[Int]): Array[BDM[Double]] = {
    // 初始化偏置向量b
    // weights and weight momentum
    // dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
    val n = size.length
    val rbm_b = ArrayBuffer[BDM[Double]]()
    for (i <- 1 to n - 1) {
      //val d1 = BDM.zeros[Double](size(i - 1), 1)
      val d1=BDM.rand(size(i-1), 1, new Gaussian(0,0.01))
      rbm_b += d1
    }
    rbm_b.toArray
  }

  /**
   * 初始化偏置向量vb
   * 初始化为0
   */
  def Initialvb(size: Array[Int]): Array[BDM[Double]] = {
    // 初始化偏置向量b
    // weights and weight momentum
    // dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);
    val n = size.length
    val rbm_vb = ArrayBuffer[BDM[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = BDM.zeros[Double](size(i - 1), 1)
      rbm_vb += d1
    }
    rbm_vb.toArray
  }

  /**
   * 初始化偏置向量c
   * 初始化为0
   */
  def Initialc(size: Array[Int]): Array[BDM[Double]] = {
    // 初始化偏置向量c
    // weights and weight momentum
    // dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
    val n = size.length
    val rbm_c = ArrayBuffer[BDM[Double]]()
    for (i <- 1 to n - 1) {
      //val d1 = BDM.zeros[Double](size(i), 1)
      val d1=BDM.rand(size(i), 1, new Gaussian(0,0.01))
      rbm_c += d1
    }
    rbm_c.toArray
  }

  /**
   * 初始化偏置向量vc
   * 初始化为0
   */
  def Initialvc(size: Array[Int]): Array[BDM[Double]] = {
    // 初始化偏置向量c
    // weights and weight momentum
    // dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    val n = size.length
    val rbm_vc = ArrayBuffer[BDM[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = BDM.zeros[Double](size(i), 1)
      rbm_vc += d1
    }
    rbm_vc.toArray
  }
  // Gaussian variance
  def InitVar(size: Array[Int]):BDM[Double]=BDM.ones[Double](size(0), 1)
  def InitvVar(size: Array[Int]):BDM[Double]=BDM.zeros[Double](size(0),1)
  
  def samplestates(h0: BDM[Double]):BDM[Double]={
    val r1 = BDM.rand[Double](h0.rows, h0.cols)
    val a1 = h0 :> r1
    val a2 = a1.data.map { f => if (f == true) 1.0 else 0.0 }
    val a3 = new BDM(h0.rows, h0.cols, a2)
    a3
  }

  /**
   * Gibbs采样
   * X = double(1./(1+exp(-P)) > rand(size(P)));
   */
  def sigmrnd(P: BDM[Double]): BDM[Double] = {
    val s1 = 1.0 / (Bexp(P * (-1.0)) + 1.0)
    val r1 = BDM.rand[Double](s1.rows, s1.cols)
    val a1 = s1 :> r1
    val a2 = a1.data.map { f => if (f == true) 1.0 else 0.0 }
    val a3 = new BDM(s1.rows, s1.cols, a2)
    a3
  }

  /**
   * Gibbs采样
   * X = double(1./(1+exp(-P)))+1*randn(size(P));
   */
  def sigmrnd2(P: BDM[Double]): BDM[Double] = {
    val s1 = 1.0 / (Bexp(P * (-1.0)) + 1.0)
    val r1 = BDM.rand[Double](s1.rows, s1.cols)
    val a3 = s1 + (r1 * 1.0)
    a3
  }

  /**
   * sigm激活函数
   * X = 1./(1+exp(-P));
   */
  def sigm(matrix: BDM[Double]): BDM[Double] = {
    val s1 = 1.0 / (Bexp(matrix * (-1.0)) + 1.0)
    s1
  }

  /**
   * tanh激活函数
   * f=1.7159*tanh(2/3.*A);
   */
  def tanh_opt(matrix: BDM[Double]): BDM[Double] = {
    val s1 = Btanh(matrix * (2.0 / 3.0)) * 1.7159
    s1
  }

}
