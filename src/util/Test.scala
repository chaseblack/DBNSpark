package util

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import breeze.linalg.DenseMatrix
import breeze.linalg.{DenseMatrix => BDM}
import DBN.DBN
import neuralnet.NeuralNet
import breeze.linalg.{DenseMatrix => BDM}

object Test {
  def main(args:Array[String]){
    org.apache.log4j.PropertyConfigurator.configure("/home/chase/runnable/spark-2.0.2-bin-hadoop2.7/conf/log4j.properties");
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local")
    val sc = new SparkContext(conf)
    val rdd1 = sc.parallelize(Array(1,2,3))
    val rdd2 = sc.parallelize(Array(1,2,3))
    val u=rdd1.union(rdd2)
    for(x<-u.collect())print(x+" ")
    
    val label=Array("井号","渗透性类型","岩性","井型","抽油机机型","电机型号","泵况分析结果","对标结果")
    val y=Array("对标结果")
    val x=Array("造斜点深度(m)","生产时间(h)","日产液(t)","日产油(t)","日产水(t)","日产气(m3)","含水率(%)","油压(Mpa)","套压(Mpa)","动液面(m)","泵径(mm)","泵深(m)","冲程(m)","冲次(n/min)","泵效(%)","额定载荷(kN)","悬点载荷最大值(kN)","悬点载荷最小值(kN)","电机额定功率(kW)","平衡度","日耗电量(kW·h)","输入功率(kW)","地面效率(%)","井下效率(%)","系统效率(%)","沉没度(m)","抽油机负载率(%)","电机功率利用率(%)","百米吨液耗电(kW.h)","K1","K2","节能限定值","节能评价值")
    println(label.length+" "+x.length)
  }
  
}