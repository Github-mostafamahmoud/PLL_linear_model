`timescale 1ps/1ps
parameter int n=23;
module Quant_1b(input signed [n:0] alpha, input signed [n:0] u, input    clk,output  reg out );
parameter signed TH;
 always@(posedge clk)begin
#4
if (alpha==0)
out<=0;
else begin
if (u>=TH)
out<=1;
else
out<=0;
end
end
initial begin
out=0;
end
endmodule
