package examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import breeze.linalg.DenseMatrix
import breeze.linalg.{DenseMatrix => BDM}
import DBN.DBN
import neuralnet.NeuralNet
import breeze.linalg.{DenseVector => BDV}
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix

import scala.collection.mutable.ArrayBuffer


object DBNApp {
    def extract(dataWithHead:RDD[String],index:(Int,Array[Int]))={
      val head=dataWithHead.first()
      val dataNoHead=dataWithHead.filter { x => x!=head}//remove the head,e.g. the first line
      val y_index=index._1// e.g. y_index=40 means 对标结果
      val x_index=index._2//e.g. x_index=Array(5,6,7) means 日产液(t), 日产油(t), 日产水(t)
      val num_vislayer=x_index.length
      val num_labellayer=dataNoHead.first().split(" ").apply(y_index).split(",").length
      val train_d=dataNoHead.map{
        line =>
          val featureStrArray = line.split(" ")
          val y=featureStrArray.apply(y_index).split(",").map { x => x.toDouble }
          val x=ArrayBuffer[Double]()
          for(i<-0 to num_vislayer-1){
            val feature_index=x_index.apply(i)// e.g. feature_index=5, then it is 日产液(t)
            val value=featureStrArray.apply(feature_index).toDouble// then value may be 22.83
            x+=value
          }
          /***********************ATTENTION**************************************
           * Mathematically, each sample in the training set should be a DenseVector. 
      		 * However, in this code each sample is wrapped in a DenseMatrix. Why? I personally
           * think this is because multiplication between a vector and a matrix in Breeze is tricky
           * from my prior experience on Breeze.
           * 
           * BDM(1,x.length,x) is actually a vector, mathematically. the row number is 1, 
           * the col number is x.length, and the vector data are from x itself. 
           * Therefore I denote those BDMs that are actually vectors.**/
          (new BDM(1,y.length,y), new BDM(1,x.toArray.length,x.toArray))
          }
      (num_vislayer,train_d,num_labellayer)
      /*
      val y=label.map {
        line =>
          val f1 = line.split(",")
          val y = f1.map(f =>f.toDouble)
          y
          //new BDM(1,y.length,y)
          }
      //val u=y.union(x)
      val u=x++y
      //val u=x
      val u0=u.collect()(0)
      for(e<-u0)print(e+" ")
      println()
      val u1=u.collect()(1)
      for(e<-u1)print(e+" ")
      println()
      val u2=u.collect()(2)
      for(e<-u2)print(e+" ")
      println()
      val train_d=y.union(x).map { f => 
        val y=f.slice(0, num_labellayer-1) 
        val x=f.slice(num_labellayer, f.length) 
        (new BDM(1,y.length,y),new BDM(1,x.length,x))
        }*/
      }
   
  
    
  val label_global=Map("井号"->0,"渗透性类型"->1,"岩性"->2,"井型"->3,"抽油机机型"->19,"电机型号"->23,"泵况分析结果"->31,"对标结果"->40)
  val x_global=Map("造斜点深度(m)"->4,"生产时间(h)"->5,"日产液(t)"->6,"日产油(t)"->7,"日产水(t)"->8,"日产气(m3)"->9,"含水率(%)"->10,"油压(Mpa)"->11,"套压(Mpa)"->12,"动液面(m)"->13,"泵径(mm)"->14,"泵深(m)"->15,"冲程(m)"->16,"冲次(n/min)"->17,"泵效(%)"->18,"额定载荷(kN)"->20,"悬点载荷最大值(kN)"->21,"悬点载荷最小值(kN)"->22,"电机额定功率(kW)"->24,"平衡度"->25,"日耗电量(kW·h)"->26,"输入功率(kW)"->27,"地面效率(%)"->28,"井下效率(%)"->29,"系统效率(%)"->30,"沉没度(m)"->32,"抽油机负载率(%)"->33,"电机功率利用率(%)"->34,"百米吨液耗电(kW.h)"->35,"K1"->36,"K2"->37,"节能限定值"->38,"节能评价值"->39)
  def getAttributeIndex(param_y:String, param_x:Array[String])={
    val y_index=label_global.get(param_y).get
    val x_index=ArrayBuffer[Int]()
    //bian li one by one
    for(i <- 0 to param_x.length-1){
      //println(param_x(i)+" "+x_global.get(param_x(i)).get)
      x_index+=(x_global.get(param_x(i)).get) 
    }
    (y_index,x_index.toArray)
  }
  
  def PCA(sc:SparkContext,train_d:RDD[(DenseMatrix[Double],DenseMatrix[Double])])={
    val label=train_d.map{x=>
      x._1
    }
     val feature=train_d.map{x=>
      val v=Vectors.dense(x._2.data)
      v
    }
    val mat: RowMatrix = new RowMatrix(feature)
    
    println(mat.numRows()+"  "+mat.numCols())
    val pc: Matrix = mat.computePrincipalComponents(7)
    println(pc.numRows+"  "+pc.numCols)
    val projected: RowMatrix = mat.multiply(pc)//projected is put into ANN
    println(projected.numRows()+"  "+projected.numCols())
    val reduced_feature=projected.rows.map { x => x.toArray }.collect()
    val label1=label.map { x => x.data }
    val label2=label1.collect()
    val xy=new ArrayBuffer[(BDM[Double],BDM[Double])]()
    for(i <-0 to label2.length-1){
      val y=new BDM[Double](1, label2(i).length,label2(i))
      val x=new BDM[Double](1,reduced_feature(i).length,reduced_feature(i))
      val c=(y,x)
      xy+=c
    }
    val xyA=xy.toArray
    val reduced_train_d=sc.parallelize(xyA.toSeq)
    reduced_train_d
  }
  def main(args:Array[String]){
    org.apache.log4j.PropertyConfigurator.configure("/home/chase/runnable/spark-2.0.2-bin-hadoop2.7/conf/log4j.properties");
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local")
    val sc = new SparkContext(conf)
    val data_file="cleanedBPU.csv"
    val data = sc.textFile(data_file, 1).cache()
    val index=getAttributeIndex("对标结果",Array("日产液(t)","日产油(t)","日产水(t)","含水率(%)","动液面(m)","冲次(n/min)","泵效(%)","平衡度","日耗电量(kW·h)","地面效率(%)","井下效率(%)","系统效率(%)","沉没度(m)","抽油机负载率(%)"))
    val tuple=DBNApp.extract(data,index) 
    var num_vislayer=tuple._1
    var train_d =tuple._2
    
    //train_d=PCA(sc,train_d)
    //num_vislayer=train_d.first()._2.cols
      
    val num_labellayer=tuple._3
    val opts = Array(100.0,20.0,1)//batchsize, numepoch, k_CD 
    val DBNmodel =new DBN().setSize(Array(num_vislayer, 4,num_labellayer)).setLayer(3).setMomentum(0.01).setAlpha(0.01).DBNtrain(train_d, opts)
    
    println("train NN......")
    val mynn = DBNmodel.dbnunfoldtonn(0)
    val nnopts = Array(100.0, 200.0, 0.0)//batchsize, numepoch,
    val NNmodel = new NeuralNet().setSize(mynn._1).setLayer(mynn._2).setActivation_function("sigm").setOutput_function("sigm").setInitW(mynn._3).NNtrain(train_d, nnopts)
    
    val deviat=DBNmodel.Gauss_deviat
    val deviatedTrainData=train_d.map{f=>
      val x=f._2
      //(f._1,x:/deviat.t)
      (f._1,x)
    }
    
    val NNforecast = NNmodel.predict(deviatedTrainData)
    val NNerror = NNmodel.Loss(NNforecast)
    println(s"NNerror = $NNerror.")
    val printf1 = NNforecast.map(f => (f.label.data, f.predict_label.data)).take(925)
    val result=printf1.map{x=>
      val vec2=new BDV(x._2)
      val v2=vec2:>0.3333333
      val v22=v2.map{x=>if(x) 1.0 else 0.0}
      val vec1=new BDV(x._1)
      val r=vec1.t*v22
      r
      }
    var accum=0.0;
    for(e<-result){
      accum=accum+e
    }
    println(accum/result.length)
    
    
   /*
    println("预测结果——实际值：预测值")
    for (i <- 0 until printf1.length){ 
      for(e<-printf1(i)._1)print(e+" ")
      for(e<-printf1(i)._2)print(e+" ")
      println()
      }
    */
  }
  
  
  
  
    /*
    val numExamples = train_d.count()
    println(s"numExamples = $numExamples.")
    println(mynn._2)
    for (i <- 0 to mynn._1.length - 1) {
      print(mynn._1(i) + "\t")
    }
    println()
    println("mynn_W1")
    val tmpw1 = mynn._3(0)
    for (i <- 0 to tmpw1.rows - 1) {
      for (j <- 0 to tmpw1.cols - 1) {
        print(tmpw1(i, j) + "\t")
      }
      println()
    }
    */
}