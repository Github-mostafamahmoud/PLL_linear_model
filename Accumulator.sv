module accum #(parameter int n=9)(input signed [n:0] alpha, input CLK, output reg  out,input signed[n:0] QNC);

reg signed [n+1:0] summer;

always @(posedge CLK ) begin
summer=summer+alpha;
if(summer>=(2**(n))) begin
summer=0;
out=1;
end
else
out=0;
end
 initial begin
summer=0;
out=0;
end
endmodule;
