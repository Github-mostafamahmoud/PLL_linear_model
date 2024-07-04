`timescale 1ps/1ps
//parameter int n=23; 
module Sigma_Delta_internal #(parameter integer n = 9) (input signed [n:0] alpha,input CLK , output reg  out,output signed  [n:0] QNC );

reg signed [n:0] sum1;
reg signed [n:0] Prev1st;
reg signed [n:0] Prev2nd;
reg signed [n:0]int1; 
reg signed [n:0] sum2;
reg signed [n:0] int2;
wire signed [n:0] neg_int2;
reg signed [n:0] prev;
real division;
assign QNC=int1;
assign neg_int2=-int2;
always @(posedge CLK)begin

//x generation
if (out==1)
sum1=alpha-(2**(25));
else
sum1=alpha;
//integration1
Prev1st = Prev1st+sum1;
int1 =Prev1st;

//z generation
if (out==1)
sum2=int1-(2**(25));
else
sum2=int1;

division=real'(alpha)/real'((2**(n-1)));

//integration2
Prev2nd = Prev2nd+sum2;
int2 =Prev2nd;
//QUANTAIZER
if (alpha==0)
out=0;
else begin
if (int2>0)//TH)
out=1;
else
out=0;
end
end

initial begin 
sum1=0;
sum2=0;
prev=0;
Prev2nd=0;
Prev1st=0;
out=1;
end
endmodule
