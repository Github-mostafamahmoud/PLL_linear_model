`timescale 1ns/1ps
module clkdiveven (vco_out,x,clk_out);
input reg vco_out;
input reg  x;
output reg clk_out;
parameter integer N;
integer  y ;
logic z;
logic L;
logic K;
integer pos_counter=0;
integer neg_counter=0;
integer pos_counter1=0;
integer neg_counter1=0;
integer Divisor_by_2;
assign K = (y) == 0;
assign Divisor_by_2=(N+x)/2;
assign y = (N+x)%2;
assign z = ( pos_counter > (Divisor_by_2)) || ( neg_counter > (Divisor_by_2));
assign L= pos_counter >= (Divisor_by_2);
always @ (posedge vco_out)
begin
//Divisor_by_2=(N+x)/2;
pos_counter=(pos_counter+1)%(N+x);
pos_counter1<=(pos_counter1+1)%(N+x);
end
always@(negedge vco_out)
begin
neg_counter=(neg_counter+1)%(N+x);
neg_counter1<=(neg_counter1+1)%(N+x);
end
//always@(posedge vco_out or negedge x)
always@(vco_out)
begin
if (N%2==0) begin 
if (x==1) begin
 if (K)//means N is even
clk_out = L; //? 1'b1 : 1'b0;
else //(N+x) is odd
clk_out =  ( neg_counter > (Divisor_by_2)) ? 1'b1 : 1'b0; 
end
else begin
if (K)//means N is even
clk_out = L; //? 1'b1 : 1'b0;
else //(N+x) is odd
clk_out = z; //? 1'b1 : 1'b0; 
end 
 end 
else begin 
if (K)//means N is even
clk_out <= ( neg_counter1 >= (Divisor_by_2)) ? 1'b1 : 1'b0;
else //(N+x) is odd
clk_out <= (( pos_counter1 > (Divisor_by_2)) || ( neg_counter1 > (Divisor_by_2))) ? 1'b1 : 1'b0; 
end
end 
endmodule