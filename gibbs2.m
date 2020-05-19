function xx = gibbs2(N, pau); % Gibbs sampling demo

C=[1 0.98; 0.98 1];
Ci=inv(C);
m=[0 0];
clf;

x=[-2 2];

if pau ~= 0
  subplot(211)
  plot_gaussian(4*C,m,2,60);
  set(gcf,'Renderer','zbuffer');
  pause;
end

xx = x;
for i=1:N
if pau ~= 0
  subplot(211)
  axis([-3 3 -3 3]);
end
  xold=x;
  x(1) = -Ci(1,2)*x(2)/Ci(1,1) + randn/sqrt(Ci(1,1));
if pau ~= 0
  plot([xold(1) x(1)],[xold(2) x(2)],'-','Color',0.65*[1 1 1]);
  hold on;
end
  xold=x;
  x(2) = -Ci(1,2)*x(1)/Ci(2,2) + randn/sqrt(Ci(2,2));
if pau ~= 0
  plot([xold(1) x(1)],[xold(2) x(2)],'-','Color',0.65*[1 1 1]);
  plot(x(1),x(2),'.','MarkerSize',10);
  drawnow;
  pause(pau);
end
  xx = [xx; x];
if pau ~= 0
  subplot(212)
  plot(xx);
  end
end;
     
