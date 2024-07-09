#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:16:59 2020

@author: bruzewskis

This code is a new implementation to use the graph generation code for
identifying sets of sources in Fermi fields. It's definitely going to steal 
code from the six other versions of this code I've written over time, but 
this one will be the best, I promise.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from tqdm import tqdm


from scipy.optimize import curve_fit

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, Angle

from catloader import reader

def norm_fermi_rad(sources, fermi):
    """
    Calculates the normalized radius of a particular source. Note that the
    normalization is relative to the 'effective radius of the error ellipse'
    corresponding the the 95% confidence interval. 
    
    Args
    =======
    sources (Table) - Assumed to be an astropy table, but can be an object
        which is key accessible. Must contain columns 'RA' and 'Dec'
        
    fermi (dict) - A dictionary containing the information on the fermi source
        which lets us generate the normalization radius.
    
    Returns
    =======
    normalized (array) - An array of normalized radii for all the input 
        sources. Note that 1 is equivalent to the ellipse edge.
    
    
    Raises
    =======
    None
    """
    
    # Turn all coordinates into SkyCoord objects
    scf = SkyCoord(fermi['RAJ2000'], fermi['DEJ2000'], unit=['deg','deg'])
    scs = SkyCoord(sources['RA'], sources['Dec'], unit=['deg','deg'])
    
    # Determine separation properties
    seps = scf.separation(scs).deg
    posangs = scf.position_angle(scs).deg
    
    # Define Fermi properties we need
    fa = fermi['Conf_95_SemiMajor']
    fb = fermi['Conf_95_SemiMinor']
    heading = np.deg2rad( posangs - fermi['Conf_95_PosAng'] )
    
    # Calculate normalized distance
    psi = 1/np.sqrt( (np.cos(heading)/fa)**2 + (np.sin(heading)/fb)**2 )
    normalized = seps/psi
    
    return seps, normalized, posangs
    
    
def show_sources_field(sources, fermi):
    
    # Shortcut parameters
    x0, y0 = fermi['RAJ2000'], fermi['DEJ2000']
    fmaj = fermi['Conf_95_SemiMajor']
    fmin = fermi['Conf_95_SemiMinor']
    fpa = fermi['Conf_95_PosAng']
    pd = np.deg2rad(90-fpa)
    
    # Centering parameters
    a = np.linspace(0,2*np.pi)
    xs0 = fmaj*np.cos(a)*np.cos(pd) - fmin*np.sin(a)*np.sin(pd)
    ys0 = fmaj*np.cos(a)*np.sin(pd) + fmin*np.sin(a)*np.cos(pd)
    
    # Corrected plot
    corr = np.cos(np.deg2rad(y0))
    # plt.figure(figsize=(5,5))
    plt.plot(xs0/corr+x0, ys0+y0, 'k-')
    
    for survey in np.unique(sources['survey']):
        # Get survey subtable
        srcs = sources[sources['survey']==survey]
        
        # Plot if we have points
        if len(srcs)>0:
            plt.errorbar(srcs['RA'], srcs['Dec'], 
                         srcs['RA_Err'], srcs['Dec_Err'], marker='.', 
                         ls='none',label=survey, zorder=5)
    
    # Set the scaling
    plt.xlim(x0-1.5*fmaj/corr, x0+1.5*fmaj/corr)
    plt.ylim(y0-1.5*fmaj, y0+1.5*fmaj)
    plt.title(fermi['Source_Name'])
    plt.xlabel('RA [deg]')
    plt.ylabel('Dec [deg]')
    if len(sources)>0:
        plt.legend()
    plt.grid()

def find_interior_sources(catalogs, fermi):
    
    small_cats = []
    for name, cat in catalogs.items():
        seps, norm_rad, posangs = norm_fermi_rad(cat, fermi)
        
        # Track some useful columns in a smaller catalog
        in_ellipse = norm_rad <= 1
        small_cat = cat[in_ellipse]
        small_cat['phys_rad'] = seps[in_ellipse]
        small_cat['norm_rad'] = norm_rad[in_ellipse]
        small_cat['pos_angs'] = posangs[in_ellipse]
        small_cats.append( small_cat )
    
    interior_sources = vstack(small_cats, metadata_conflicts='silent')  
        
    # Wrap Right Ascension to be safe
    good_wrap = (fermi['RAJ2000'] + 180)%360
    wrapped_ra = Angle(interior_sources['RA']).wrap_at(f'{good_wrap}d').deg
    interior_sources['RA'] = wrapped_ra
    interior_sources['RA'].unit = 'deg'
    
    return interior_sources

def find_connections(sources):
    
    # Define some shorthand names
    ra = sources['RA']
    rae = sources['RA_Err']
    dc = sources['Dec']
    dce = sources['Dec_Err']
    
    # Find pairs
    pairs = []
    for i in range(len(sources)):
        for j in range(i, len(sources)):
            # PyBDSF does the linking inside catalogs
            if sources['survey'][i]==sources['survey'][j]:
                continue
            
            # Calculate our DeRuitter distance metric
            cos_term = np.cos(np.deg2rad((dc[i]+dc[j])/2))**2
            RA_NUM = (ra[i]-ra[j])**2 * cos_term
            RA_DEN = rae[i]**2+rae[j]**2
            DC_NUM = (dc[i]-dc[j])**2
            DC_DEN = dce[i]**2+dce[j]**2
            DR = np.sqrt( RA_NUM/RA_DEN + DC_NUM/DC_DEN )
            
            # If less than this, keep it
            r_p = np.sqrt(2*np.log(1/1e-3)) #miss 1 in 1000
            if DR < r_p:
                pairs.append([i,j])
            
    return pairs

def network_sources(cat, con, plot=False, fermi=None, find=None):
    
    G = nx.Graph()
    colors = []
    
    # Build a colormap
    colormap = {}
    surveys = np.unique(cat['survey'])
    for i in range(len(surveys)):
        colormap[ surveys[i] ] = 'C'+str(i)
        
    # For plotting, need this or it borks, basically re-re-rewrap
    gw = (fermi['RAJ2000']+180)%360
    fermi['RAJ2000'] = Angle(fermi['RAJ2000'], unit='deg').wrap_at(f'{gw}d').deg
    
    colors = []
    pos_init = {}
    # Add all sources to graph
    for i in range(len(cat)):
        G.add_node(i)
        colors.append( colormap[ cat['survey'][i] ] )
        
        # Set up positions if we need to do some plotting
        pos_init[i] = ( (cat['RA'][i]-fermi['RAJ2000'])/fermi['Conf_95_SemiMajor'], 
                        (cat['Dec'][i]-fermi['DEJ2000'])/fermi['Conf_95_SemiMajor'] )
        
    # Use USID to create same-catalog connections
    new_usids = [ cat['survey'][i]+str(cat['USID'][i]) for i in range(len(cat))]
    uni, cnt = np.unique(new_usids, return_counts=True)
    for i in range(len(uni)):
        if cnt[i]>1:
            # If there's a pair/group
            uinds = np.where(np.array(new_usids==uni[i]))[0]
            
            # Create all the links
            for i in range(len(uinds)):
                for j in range(i+1,len(uinds)):
                    G.add_edge(uinds[i], uinds[j])
                
    # Add each of our inter-catalog connections
    for pair in con:
        G.add_edge(*pair)
    
    # Break graph down into groups
    subgraphs = list(nx.connected_components(G))
    
    # Visualize stuff
    if plot:
        fig = plt.figure(figsize=(10.66666,6), dpi=180)
        ax1 = plt.subplot2grid((1,2),(0,0))
        show_sources_field(cat, fermi)
        
        # Show graph
        s0 = 100 * np.sqrt(( np.exp(-(len(subgraphs)-1)/2) + 1 ))
        k0 = s0 / 120
        ax2 = plt.subplot2grid((1,2),(0,1))
        pos = nx.spring_layout(G, 
                               pos=pos_init if len(pos_init)>0 else None,
                               k = k0,
                               iterations=50)
        nx.draw_networkx(G, pos=pos, 
                         with_labels=False, 
                         node_color=colors, 
                         node_size=s0)
        
        plt.axis('off')
        plt.xlim(-1.1,1.1)
        plt.ylim(-1.1,1.1)
        plt.tight_layout()
        fname = fermi['Source_Name'][5:]
        
        # Make directories and save the relevant info
        sdir = f'outputs/plots/{fname}'
        os.makedirs(sdir, exist_ok=True)
        sname = f'{sdir}/{fname}'
        sp = open(f'{sname}.csv', 'w')
        sp.write('num,ra,dec\n')
        
        for i in range(len(subgraphs)):
            sgi = np.array(list(subgraphs[i]))
            
            xA = np.mean(cat['RA'][sgi])
            yA = np.mean(cat['Dec'][sgi])
            
            xB = np.mean([ pos[s][0] for s in sgi ])
            yB = np.mean([ pos[s][1] for s in sgi ])
            
            oA = 3*fermi['Conf_95_SemiMajor']/100
            oB = 2/50
            ax1.text(xA+oA, yA+oA, i+1)
            ax2.text(xB+oB, yB+oB, i+1, bbox=dict(facecolor='w', alpha=0.5, boxstyle='round,pad=0.25'))
        
            sp.write(f'{i+1}, {xA}, {yA}\n')
        sp.close()
        
        plt.savefig(f'{sname}.pdf')
        plt.savefig(f'{sname}.png')
        # plt.show()
        plt.close(fig)
    
    return subgraphs

def simplepower(nu, amp, alpha):
    return amp * nu**alpha

def make_name(ra, dec):
    sc = SkyCoord(ra, dec, unit=['deg','deg'])
    
    ras = sc.ra.to_string('h', sep='', pad=True)[:9]
    dcs = sc.dec.to_string(sep='', pad=True, alwayssign=True)[:9]
    
    return 'MSC J' + ras + dcs

def process(dat, id_list, fermi):
    
    new = Table()
    
    cols = ['RA', 'RA_Err', 'Dec', 'Dec_Err', 'Freq', 'Peak_Flux', 
            'Peak_Flux_Err', 'Int_Flux', 'Int_Flux_Err', 'phys_rad', 
            'norm_rad']
    
    # For line in F357
    for group in id_list:
        
        # Get members of that line
        gids = list(group)
        gtab = dat[gids]
        
        # For ease of use
        surveys = np.unique(gtab['survey'])
        
        # Line info
        out = {}
        out['fermi_name'] = [fermi['Source_Name']]
        out['fermi_class'] = [fermi['class_new']]
        out['surveys'] = [set(surveys)]
        
        # Generate names
        name_map = {}
        for survey in surveys:
            sgtab = gtab[gtab['survey']==survey]
            
            name_map[survey] = sgtab['Name'][0]
        out['name'] = [name_map]

        
        # Generate all the column dicts
        for col in cols:
            smap = {}
            
            for survey in  surveys:
                sgtab = gtab[gtab['survey']==survey]
                
                smap[survey] = np.nanmean(sgtab[col])
                
            out[col] = [smap]
        
        # Append new line
        new = vstack([new, Table(out)])
    
    return new

def spectrum(freq, freq0, flux0, a0, a1):
    x = np.log(freq/freq0)
    S = flux0 * np.exp( a0 * x + a1 * x**2 )
    return S

def find_alphas(dat):
    
    # General setup for these columns
    length = len(dat)
    dat['ref_freq']     = np.full(length, np.nan)
    dat['ref_flux']     = np.full(length, np.nan)
    dat['ref_flux_err'] = np.full(length, np.nan)
    dat['alpha']        = np.full(length, np.nan)
    dat['alpha_err']    = np.full(length, np.nan)
    dat['curve']        = np.full(length, np.nan)
    dat['curve_err']    = np.full(length, np.nan)
    
    # Go find them all
    for i in range(len(dat)):
        
        surveys = dat['surveys'][i]
        
        freq = np.array([ dat['Freq'][i][s] for s in surveys ])
        flux = np.array([ dat['Int_Flux'][i][s] for s in surveys ])
        ferr = np.array([ dat['Int_Flux_Err'][i][s] for s in surveys ])
        
        mid_freq = np.sqrt(np.max(freq)*np.min(freq))
        mid_flux = np.sqrt(np.max(flux)*np.min(flux))
        
        spec_ref_linear = lambda f, flux0, a0 : spectrum(f, mid_freq, 
                                                         flux0, a0, 0)
        spec_ref_curved = lambda f, flux0, a0, a1 : spectrum(f, mid_freq, 
                                                             flux0, a0, a1)
        
        # Fit in different ways
        num_unique_freqs = len(np.unique(freq))
        if num_unique_freqs == 2:
            # Might have a scenario where we have multiple values at a
            # particular frequency, so we need to make sure that at each
            # frequency we have an average
            ufreq = np.unique(freq)
            uflux = np.array([ np.sum(flux[freq==f]) for f in ufreq ])
            uferr = np.array([ np.sqrt(np.sum(ferr[freq==f]**2 * flux[freq==f]**2)) for f in ufreq ])
            
            # Calculate the values as best we can
            denom = np.log(ufreq[1]/ufreq[0])
            if np.any(uflux==0):
                print('~'*10, i)
                print(ufreq)
                print(uflux)
                print(uferr)
            alpha_calc = np.log(uflux[1]/uflux[0]) / denom
            err_numerator = (uferr[0]/uflux[0])**2+(uferr[1]/uflux[1])**2
            alpha_calc_err = np.sqrt(err_numerator / abs(denom))
            
            mid_flux_calc = spectrum(mid_freq, ufreq[1], uflux[1], alpha_calc,0)
            mid_flux_calc_err = np.sqrt( (uferr[1]/uflux[1])**2 
                                        + denom**2 * alpha_calc_err**2)
            
            # Assign to Table
            dat['ref_freq'][i]      = mid_freq
            dat['ref_flux'][i]      = mid_flux_calc
            dat['ref_flux_err'][i]  = mid_flux_calc_err
            dat['alpha'][i]         = alpha_calc
            dat['alpha_err'][i]     = alpha_calc_err
            
        elif num_unique_freqs == 3:
            # Perform some fits linearly
            p0 = [mid_flux, -0.5]
            popt, pcov = curve_fit(spec_ref_linear, freq, flux, sigma=ferr, p0=p0)
            perr = np.sqrt(np.diag(pcov))
            
            # Assign to Table
            dat['ref_freq'][i]      = mid_freq
            dat['ref_flux'][i]      = popt[0]
            dat['ref_flux_err'][i]  = perr[0]
            dat['alpha'][i]         = popt[1]
            dat['alpha_err'][i]     = perr[1]
            
        elif num_unique_freqs > 3:
            # Perform some fits with curve
            p0 = [mid_flux, -0.5, 0]
            popt, pcov = curve_fit(spec_ref_curved, freq, flux, sigma=ferr, p0=p0)
            perr = np.sqrt(np.diag(pcov))
            
            # Assign to Table
            dat['ref_freq'][i]      = mid_freq
            dat['ref_flux'][i]      = popt[0]
            dat['ref_flux_err'][i]  = perr[0]
            dat['alpha'][i]         = popt[1]
            dat['alpha_err'][i]     = perr[1]
            dat['curve'][i]         = popt[2]
            dat['curve_err'][i]     = perr[2]
    
    return dat
                
def organize(dat):
    
    lines = []
    for i in range(len(dat)):
        surveys = dat['surveys'][i]
        
        # Calculate a good RA and DEC for the object
        x = np.array([ dat['RA'][i][s] for s in surveys ])
        xe = np.array([ dat['RA_Err'][i][s] for s in surveys ])
        xw = 1/xe**2
        y = np.array([ dat['Dec'][i][s] for s in surveys ])
        ye = np.array([ dat['Dec_Err'][i][s] for s in surveys ])
        yw = 1/ye**2
        
        # Take a weighted average and a weighted standard error of mean
        xwm = np.sum(x*xw)/np.sum(xw)
        xscale = np.sqrt(np.sum(xw**2)/np.sum(xw)**2)
        xwe = np.std(x)*xscale if len(surveys)>1 else xe[0]
        
        ywm = np.sum(y*yw)/np.sum(yw)
        yscale = np.sqrt(np.sum(yw**2)/np.sum(yw)**2)
        ywe = np.std(y)*yscale if len(surveys)>1 else ye[0]
        
        # Use that info to make a name
        name = make_name(xwm, ywm)
        
        # Write that down write that down
        line = {'Name':         name,
                'FermiName':    dat['fermi_name'][i],
                'FermiClass':   dat['fermi_class'][i],
                'RA':           xwm,
                'RAErr':        xwe,
                'Dec':          ywm,
                'DecErr':       ywe}
        
        # Grab names
        for survey in surveys:
            line['Name_'+survey.upper()] = dat['name'][i][survey]
            
        lines.append(line)
    
    # Make the fill more appropriate
    ndat = Table(lines).filled('')
    
    # Apply a common wrap to the entire data set
    ras = Angle(ndat['RA'], unit='deg').wrap_at('360d').deg
    ndat['RA'] = ras
    
    # Add math we did
    ndat['RefFreq'] = dat['ref_freq']
    ndat['RefFlux'] = dat['ref_flux']
    ndat['RefFluxErr'] = dat['ref_flux_err']
    ndat['Alpha'] = dat['alpha']
    ndat['AlphaErr'] = dat['alpha_err']
    ndat['Curve'] = dat['curve']
    ndat['CurveErr'] = dat['curve_err']
    
    # Reorder things
    order = ['Name', 'FermiName', 'FermiClass', 'RA', 'RAErr', 'Dec', 'DecErr']
    order += sorted([ c for c in ndat.colnames if 'Name_' in c ])
    order += ['RefFreq', 'RefFlux', 'RefFluxErr', 'Alpha', 'AlphaErr', 
              'Curve', 'CurveErr']
    ndat = ndat[order]
    
    # Fix some units
    umap = {'deg': ['RA', 'RAErr', 'Dec', 'DecErr'],
            'Hz': ['RefFreq'],
            'mJy': ['RefFlux', 'RefFluxErr']}
    for u in umap:
        for col in umap[u]:
            ndat[col].unit = u
            
    # Add some descriptions
    meta = {'Name':         {'Description': 'Generated positional name',                        
                             'format': '{}'},
            'FermiName':    {'Description': 'Fermi field which contains this source',          
                             'format': '{}'},
            'FermiClass':   {'Description': 'Fermi object classification',
                             'format': '{}'},
            'RA':           {'Description': 'Weighted average of Right Ascension (J2000)',      
                             'format': '{:.5f}'},
            'RAErr':        {'Description': 'Weighted error on average RA',                     
                             'format': '{:.5e}'},
            'Dec':          {'Description': 'Weighted average of Declination (J2000)',          
                             'format': '{:.5f}'},
            'DecErr':       {'Description': 'Weighted error on average Dec',                    
                             'format': '{:.5e}'},
            'RefFreq':      {'Description': 'Reference frequency at the log-midpoint of data',  
                             'format': '{:.4e}'},
            'RefFlux':      {'Description': 'Fit flux value at RefFreq',                        
                             'format': '{:.3f}'},
            'RefFluxErr':   {'Description': 'Fit uncertianty on RefFlux',                       
                             'format': '{:.3e}'},
            'Alpha':        {'Description': 'Fit spectral index, positive convention',          
                             'format': '{:.3f}'},
            'AlphaErr':     {'Description': 'Fit uncertianty on Alpha',                         
                             'format': '{:.3e}'},
            'Curve':        {'Description': 'Fit spectral curvature',                           
                             'format': '{:.3f}'},
            'CurveErr':     {'Description': 'Fit uncertianty on Curve',                         
                             'format': '{:.3e}'}}
            
    for col in meta:
        ndat[col].description = meta[col]['Description']
        ndat[col].format = meta[col]['format']
    
    return ndat

def search_field(fermi, cat_dict, verbose=True):
    # Searches a single field `fermi` for any multi-frequency sources
    
    # Get sources inside elliose
    inside_sources = find_interior_sources(cat_dict, fermi)
    if verbose: 
        ncats = len(np.unique(inside_sources['survey']))
        print('Found', len(inside_sources), 'interior sources from', ncats, 'catalogs')
    
    # Identify source connections
    conns = find_connections(inside_sources)
    if verbose:
        print('Identified', len(conns), 'inter-catalog connections')
    
    # Separate into sources with networkx
    group_id = network_sources(inside_sources, conns, 
                               plot=True, fermi=fermi)
    
    # Stack together in a nice way
    sources = process(inside_sources, group_id, fermi)
    if verbose:
        print('Connected into', len(sources), 'MSC sources')
    
    return sources

def main():
    
    ############# GET FERMI DATA #############
    dat_fermi = Table.read('catalogs/fermi/FERMI_DR3_20210806.fits', hdu=1)
    dat_fermi = dat_fermi[~np.isnan(dat_fermi['Conf_95_SemiMajor'])]
    
    # Little shortcut for accessing Fermi values
    selects = ['Source_Name', 'RAJ2000', 'DEJ2000', 'Conf_95_SemiMajor', 
                'Conf_95_SemiMinor', 'Conf_95_PosAng', 'class_new']
    fermi_dicts = [ {s:dat_fermi[s][i] for s in selects} for i in range(len(dat_fermi))]
    
    ############# GET RADIO CATALOGS #############
    cats_dir = 'catalogs/radio'
    cat_files = [ f for f in os.listdir(cats_dir) if f.endswith('.fits') ]
    cat_dict = {}
    for file in tqdm(cat_files):
        cat_name = file.replace('.fits','')
        path = os.path.join(cats_dir, file)
        cat_dict[cat_name] = reader(path)
        
    ############# LOOP THROUGH FIELDS #############
    print('\nBEGINNING ANALYSIS\n')
    field_tables = []
    for i in range(len(dat_fermi)):
        print('='*20, f'({i:04d})', fermi_dicts[i]['Source_Name'], '='*20)
        # Search field for multi-frequency sources
        sources = search_field(fermi_dicts[i], cat_dict, True)
        field_tables.append(sources)
        
    ############# GET INDEX, MAKE PRETTY, WRITE OUT #############
    all_sources = vstack(field_tables, metadata_conflicts='silent')
    dat = find_alphas(all_sources)
    odat = organize(dat)
    
    odat.write('outputs/MSC.fits', overwrite=True)

if __name__=='__main__':
    np.random.seed(10)
    main()
    # dat = pickle.load( open( "testing.p", "rb" ) )
    # odat = organize(dat)
        
        
        
        
        