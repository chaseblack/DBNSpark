package util

import java.io.StringWriter
import au.com.bytecode.opencsv.CSVWriter
import scala.collection.JavaConversions._
import java.io.FileWriter
import java.io.BufferedWriter
 
object Write2CSV extends App {
val out = new BufferedWriter(new FileWriter("/home/piyush/Desktop/employee.csv"));
val writer = new CSVWriter(out);
val employeeSchema=Array("name","age","dept")
 
val employee1= Array("piyush","23","computerscience")
 
val employee2= Array("neel","24","computerscience")
 
val employee3= Array("aayush","27","computerscience")
 
var listOfRecords= List(employeeSchema,employee1,employee2,employee3)
 
writer.writeAll(listOfRecords)
out.close()
 
}
