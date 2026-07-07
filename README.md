**Hydrogenoid Atom Engine**

An interactive tool that shows what an electron's probability cloud looks 
like around a hydrogen-like atom.

<img width="2158" height="1514" alt="E7FB65BE-7D4A-41FD-BB88-69DCE56A8B9E_1_201_a" src="https://github.com/user-attachments/assets/907b461a-fe83-4370-943a-8ebaf6723529" />


* Visualize the standard quantum mechanics of a single electron in a 
hydrogen-like atom.
* Uses real math to compute the radial and angular parts of the 
wavefunction.
* Generates a point cloud that accurately represents the probability 
density of the electron.
* Optionally shows a wireframe of the outer boundary of the cloud.

**Two viewing modes:**

* REAL mode: shows the familiar lobed shapes (like classic p and d orbital 
pictures).
* COMPLEX mode: shows the mathematically "pure" quantum states, which are 
always ring-shaped.

**Important notes:**

* The tool caps at n = 8 due to performance limits.
* Only the outermost layer of the wireframe is shown for now.
* This tool only models a single electron in an ion (like He+), not a real 
multi-electron atom.
* The gentle motion and camera drift are visual effects, not physical 
representations.

**Built with:**

* React
* Three.js
* Vite

**Running it locally:**

1. Run `npm install` to install dependencies.
2. Run `npm run dev` to start the development server.

**Deploying:**

* Connect this repo to Vercel or Netlify for a fast live link.
* Use GitHub Pages by setting up auto-publishing in your repository 
settings.


>>> Send a message (/? for help)
