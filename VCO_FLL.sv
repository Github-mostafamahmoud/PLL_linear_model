 `timescale 1ns/1fs

/////////////////////////////////////////////////////

module VCO_FLL(N,cfs,alpha);
input integer N;
parameter integer n;
input signed [n:0] alpha;
real div;
output unsigned [5:0] cfs;
parameter real Fref=150e6;
parameter real min_freq=9e9;
parameter real max_freq=11e9;
parameter real kvco=55e6;
parameter real Vmax;
real fcont,step,lock_freq,last_min,temp_freq,temp;
real range = Vmax*kvco;
real lower_limit = (range/2) - (0.1)*kvco;
real upper_limit = (range/2) + (0.1)*kvco;
real lower_limit1 = (range/2) - (0.2)*kvco;
real upper_limit1 = (range/2) + (0.2)*kvco;
integer Ncurves,i;
//define step 
assign div = N+(real'(alpha)/16);
assign step=0.0378e9;
assign last_min=max_freq-range;
assign Ncurves=(last_min-min_freq)/step;
assign lock_freq=div*Fref;
 
always@(*)
begin
for(i=0;i<Ncurves;i=i+1)begin
temp_freq=min_freq+(i*step);
temp=temp_freq-lock_freq;
	if(temp<0)begin
	  temp=-temp; end
if (temp>lower_limit && temp<upper_limit)begin
//cfs[5:0]<=i;
break;
if(temp>lower_limit1 && temp<upper_limit1) begin
break;
end
end
end
end
assign cfs = i;
endmodule